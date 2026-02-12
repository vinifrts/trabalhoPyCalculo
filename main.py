import sympy as sp
import numpy as np

trechos = [
    {"nome": "i - reta horizontal", "extensao": 5, "tempo_metro": 0.8},
    {"nome": "ii - buraco", "extensao": 2, "tempo_metro": 0.4},
    {"nome": "iii - reta horizontal", "extensao": 3, "tempo_metro": 0.8},
    {"nome": "iv - rampa subindo", "extensao": 3, "tempo_metro": 1.0},
    {"nome": "v - reta horizontal", "extensao": 2, "tempo_metro": 0.8},
    {"nome": "vi - rampa descendo", "extensao": 4, "tempo_metro": 0.6},
    {"nome": "vii - reta horizontal", "extensao": 3, "tempo_metro": 0.8},
    {"nome": "viii - salto", "extensao": 4, "tempo_metro": 0.2},
    {"nome": "ix - reta horizontal", "extensao": 5, "tempo_metro": 0.8},
]

tempo_acum = [0]
espaco_acum = [0]

print("tabela de tempos e espacos acumulados")
print(f"{'trecho':<25} {'tempo (s)':<12} {'espaco (m)':<12}")
print(f"{'inicio':<25} {0:<12.2f} {0:<12.0f}")

for trecho in trechos:
    tempo = trecho["extensao"] * trecho["tempo_metro"]
    tempo_acum.append(tempo_acum[-1] + tempo)
    espaco_acum.append(espaco_acum[-1] + trecho["extensao"])
    print(f"{trecho['nome']:<25} {tempo_acum[-1]:<12.2f} {espaco_acum[-1]:<12.0f}")

x_data = np.array(tempo_acum)
y_data = np.array(espaco_acum)

grau = 4
coef = np.polyfit(x_data, y_data, grau)

x = sp.Symbol('x')
s_x = sum(coef[i] * x**(grau-i) for i in range(len(coef)))
print("\nfuncao s(x) =", s_x)

v_x = sp.diff(s_x, x)
a_x = sp.diff(v_x, x)
print("\nvelocidade v(x) =", v_x)
print("\naceleracao a(x) =", a_x)

v_func = sp.lambdify(x, v_x, 'numpy')
a_func = sp.lambdify(x, a_x, 'numpy')

pontos_criticos_v = sp.solve(a_x, x)
x_eval = [0, tempo_acum[-1]] + [float(p) for p in pontos_criticos_v if p.is_real and 0 <= p <= tempo_acum[-1]]
velocidades = [(xi, float(v_func(xi))) for xi in x_eval]
velocidades.sort(key=lambda t: t[1], reverse=True)
print("\nvelocidade maxima:", velocidades[0][1], "m/s")
print("ocorre em t =", velocidades[0][0], "s")

da_dx = sp.diff(a_x, x)
pontos_criticos_a = sp.solve(da_dx, x)
x_eval_a = [0, tempo_acum[-1]] + [float(p) for p in pontos_criticos_a if p.is_real and 0 <= p <= tempo_acum[-1]]
aceleracoes = [(xi, float(a_func(xi))) for xi in x_eval_a]
aceleracoes.sort(key=lambda t: abs(t[1]), reverse=True)
print("\naceleracao maxima (em modulo):", aceleracoes[0][1], "m/s^2")
print("ocorre em t =", aceleracoes[0][0], "s")

d1 = 4
N = (10 + d1) / 10
equacao = v_x - N

def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    x_n = x0
    for i in range(max_iter):
        fx = f(x_n)
        dfx = df(x_n)
        if abs(dfx) < 1e-10:
            break
        x_next = x_n - fx / dfx
        if abs(x_next - x_n) < tol:
            return x_next
        x_n = x_next
    return x_n

f_newton = sp.lambdify(x, equacao, 'numpy')
df_newton = sp.lambdify(x, sp.diff(equacao, x), 'numpy')
x0 = tempo_acum[-1] / 2
resultado = newton_raphson(f_newton, df_newton, x0)
velocidade_encontrada = float(v_func(resultado))

print("\nvelocidade alvo N =", N, "m/s")
print("tempo encontrado t =", resultado, "s")
print("verificacao v(t) =", velocidade_encontrada, "m/s")
