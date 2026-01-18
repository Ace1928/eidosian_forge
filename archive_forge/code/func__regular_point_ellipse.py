from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.polytools import gcd
from sympy.sets.sets import Complement
from sympy.core import Basic, Tuple, diff, expand, Eq, Integer
from sympy.core.sorting import ordered
from sympy.core.symbol import _symbol
from sympy.solvers import solveset, nonlinsolve, diophantine
from sympy.polys import total_degree
from sympy.geometry import Point
from sympy.ntheory.factor_ import core
def _regular_point_ellipse(self, a, b, c, d, e, f):
    D = 4 * a * c - b ** 2
    ok = D
    if not ok:
        raise ValueError('Rational Point on the conic does not exist')
    if a == 0 and c == 0:
        K = -1
        L = 4 * (d * e - b * f)
    elif c != 0:
        K = D
        L = 4 * c ** 2 * d ** 2 - 4 * b * c * d * e + 4 * a * c * e ** 2 + 4 * b ** 2 * c * f - 16 * a * c ** 2 * f
    else:
        K = D
        L = 4 * a ** 2 * e ** 2 - 4 * b * a * d * e + 4 * b ** 2 * a * f
    ok = L != 0 and (not (K > 0 and L < 0))
    if not ok:
        raise ValueError('Rational Point on the conic does not exist')
    K = Rational(K).limit_denominator(10 ** 12)
    L = Rational(L).limit_denominator(10 ** 12)
    k1, k2 = (K.p, K.q)
    l1, l2 = (L.p, L.q)
    g = gcd(k2, l2)
    a1 = l2 * k2 / g
    b1 = k1 * l2 / g
    c1 = -(l1 * k2) / g
    a2 = sign(a1) * core(abs(a1), 2)
    r1 = sqrt(a1 / a2)
    b2 = sign(b1) * core(abs(b1), 2)
    r2 = sqrt(b1 / b2)
    c2 = sign(c1) * core(abs(c1), 2)
    r3 = sqrt(c1 / c2)
    g = gcd(gcd(a2, b2), c2)
    a2 = a2 / g
    b2 = b2 / g
    c2 = c2 / g
    g1 = gcd(a2, b2)
    a2 = a2 / g1
    b2 = b2 / g1
    c2 = c2 * g1
    g2 = gcd(a2, c2)
    a2 = a2 / g2
    b2 = b2 * g2
    c2 = c2 / g2
    g3 = gcd(b2, c2)
    a2 = a2 * g3
    b2 = b2 / g3
    c2 = c2 / g3
    x, y, z = symbols('x y z')
    eq = a2 * x ** 2 + b2 * y ** 2 + c2 * z ** 2
    solutions = diophantine(eq)
    if len(solutions) == 0:
        raise ValueError('Rational Point on the conic does not exist')
    flag = False
    for sol in solutions:
        syms = Tuple(*sol).free_symbols
        rep = {s: 3 for s in syms}
        sol_z = sol[2]
        if sol_z == 0:
            flag = True
            continue
        if not isinstance(sol_z, (int, Integer)):
            syms_z = sol_z.free_symbols
            if len(syms_z) == 1:
                p = next(iter(syms_z))
                p_values = Complement(S.Integers, solveset(Eq(sol_z, 0), p, S.Integers))
                rep[p] = next(iter(p_values))
            if len(syms_z) == 2:
                p, q = list(ordered(syms_z))
                for i in S.Integers:
                    subs_sol_z = sol_z.subs(p, i)
                    q_values = Complement(S.Integers, solveset(Eq(subs_sol_z, 0), q, S.Integers))
                    if not q_values.is_empty:
                        rep[p] = i
                        rep[q] = next(iter(q_values))
                        break
            if len(syms) != 0:
                x, y, z = tuple((s.subs(rep) for s in sol))
            else:
                x, y, z = sol
            flag = False
            break
    if flag:
        raise ValueError('Rational Point on the conic does not exist')
    x = x * g3 / r1
    y = y * g2 / r2
    z = z * g1 / r3
    x = x / z
    y = y / z
    if a == 0 and c == 0:
        x_reg = (x + y - 2 * e) / (2 * b)
        y_reg = (x - y - 2 * d) / (2 * b)
    elif c != 0:
        x_reg = (x - 2 * d * c + b * e) / K
        y_reg = (y - b * x_reg - e) / (2 * c)
    else:
        y_reg = (x - 2 * e * a + b * d) / K
        x_reg = (y - b * y_reg - d) / (2 * a)
    return (x_reg, y_reg)