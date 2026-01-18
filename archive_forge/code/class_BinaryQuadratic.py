from sympy.core.add import Add
from sympy.core.assumptions import check_assumptions
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.numbers import igcdex, ilcm, igcd
from sympy.core.power import integer_nthroot, isqrt
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import _sympify
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.ntheory.factor_ import (
from sympy.ntheory.generate import nextprime
from sympy.ntheory.primetest import is_square, isprime
from sympy.ntheory.residue_ntheory import sqrt_mod
from sympy.polys.polyerrors import GeneratorsNeeded
from sympy.polys.polytools import Poly, factor_list
from sympy.simplify.simplify import signsimp
from sympy.solvers.solveset import solveset_real
from sympy.utilities import numbered_symbols
from sympy.utilities.misc import as_int, filldedent
from sympy.utilities.iterables import (is_sequence, subsets, permute_signs,
class BinaryQuadratic(DiophantineEquationType):
    """
    Representation of a binary quadratic diophantine equation.

    A binary quadratic diophantine equation is an equation of the
    form `Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0`, where `A, B, C, D, E,
    F` are integer constants and `x` and `y` are integer variables.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.solvers.diophantine.diophantine import BinaryQuadratic
    >>> b1 = BinaryQuadratic(x**3 + y**2 + 1)
    >>> b1.matches()
    False
    >>> b2 = BinaryQuadratic(x**2 + y**2 + 2*x + 2*y + 2)
    >>> b2.matches()
    True
    >>> b2.solve()
    {(-1, -1)}

    References
    ==========

    .. [1] Methods to solve Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0, [online],
          Available: https://www.alpertron.com.ar/METHODS.HTM
    .. [2] Solving the equation ax^2+ bxy + cy^2 + dx + ey + f= 0, [online],
          Available: https://web.archive.org/web/20160323033111/http://www.jpr2718.org/ax2p.pdf

    """
    name = 'binary_quadratic'

    def matches(self):
        return self.total_degree == 2 and self.dimension == 2

    def solve(self, parameters=None, limit=None) -> DiophantineSolutionSet:
        self.pre_solve(parameters)
        var = self.free_symbols
        coeff = self.coeff
        x, y = var
        A = coeff[x ** 2]
        B = coeff[x * y]
        C = coeff[y ** 2]
        D = coeff[x]
        E = coeff[y]
        F = coeff[S.One]
        A, B, C, D, E, F = [as_int(i) for i in _remove_gcd(A, B, C, D, E, F)]
        result = DiophantineSolutionSet(var, self.parameters)
        t, u = result.parameters
        discr = B ** 2 - 4 * A * C
        if A == 0 and C == 0 and (B != 0):
            if D * E - B * F == 0:
                q, r = divmod(E, B)
                if not r:
                    result.add((-q, t))
                q, r = divmod(D, B)
                if not r:
                    result.add((t, -q))
            else:
                div = divisors(D * E - B * F)
                div = div + [-term for term in div]
                for d in div:
                    x0, r = divmod(d - E, B)
                    if not r:
                        q, r = divmod(D * E - B * F, d)
                        if not r:
                            y0, r = divmod(q - D, B)
                            if not r:
                                result.add((x0, y0))
        elif discr == 0:
            if A == 0:
                s = BinaryQuadratic(self.equation, free_symbols=[y, x]).solve(parameters=[t, u])
                for soln in s:
                    result.add((soln[1], soln[0]))
            else:
                g = sign(A) * igcd(A, C)
                a = A // g
                c = C // g
                e = sign(B / A)
                sqa = isqrt(a)
                sqc = isqrt(c)
                _c = e * sqc * D - sqa * E
                if not _c:
                    z = Symbol('z', real=True)
                    eq = sqa * g * z ** 2 + D * z + sqa * F
                    roots = solveset_real(eq, z).intersect(S.Integers)
                    for root in roots:
                        ans = diop_solve(sqa * x + e * sqc * y - root)
                        result.add((ans[0], ans[1]))
                elif _is_int(c):
                    solve_x = lambda u: -e * sqc * g * _c * t ** 2 - (E + 2 * e * sqc * g * u) * t - (e * sqc * g * u ** 2 + E * u + e * sqc * F) // _c
                    solve_y = lambda u: sqa * g * _c * t ** 2 + (D + 2 * sqa * g * u) * t + (sqa * g * u ** 2 + D * u + sqa * F) // _c
                    for z0 in range(0, abs(_c)):
                        if divisible(sqa * g * z0 ** 2 + D * z0 + sqa * F, _c) and divisible(e * sqc * g * z0 ** 2 + E * z0 + e * sqc * F, _c):
                            result.add((solve_x(z0), solve_y(z0)))
        elif is_square(discr):
            if A != 0:
                r = sqrt(discr)
                u, v = symbols('u, v', integer=True)
                eq = _mexpand(4 * A * r * u * v + 4 * A * D * (B * v + r * u + r * v - B * u) + 2 * A * 4 * A * E * (u - v) + 4 * A * r * 4 * A * F)
                solution = diop_solve(eq, t)
                for s0, t0 in solution:
                    num = B * t0 + r * s0 + r * t0 - B * s0
                    x_0 = S(num) / (4 * A * r)
                    y_0 = S(s0 - t0) / (2 * r)
                    if isinstance(s0, Symbol) or isinstance(t0, Symbol):
                        if len(check_param(x_0, y_0, 4 * A * r, parameters)) > 0:
                            ans = check_param(x_0, y_0, 4 * A * r, parameters)
                            result.update(*ans)
                    elif x_0.is_Integer and y_0.is_Integer:
                        if is_solution_quad(var, coeff, x_0, y_0):
                            result.add((x_0, y_0))
            else:
                s = BinaryQuadratic(self.equation, free_symbols=var[::-1]).solve(parameters=[t, u])
                while s:
                    result.add(s.pop()[::-1])
        else:
            P, Q = _transformation_to_DN(var, coeff)
            D, N = _find_DN(var, coeff)
            solns_pell = diop_DN(D, N)
            if D < 0:
                for x0, y0 in solns_pell:
                    for x in [-x0, x0]:
                        for y in [-y0, y0]:
                            s = P * Matrix([x, y]) + Q
                            try:
                                result.add([as_int(_) for _ in s])
                            except ValueError:
                                pass
            else:
                solns_pell = set(solns_pell)
                for X, Y in list(solns_pell):
                    solns_pell.add((-X, -Y))
                a = diop_DN(D, 1)
                T = a[0][0]
                U = a[0][1]
                if all((_is_int(_) for _ in P[:4] + Q[:2])):
                    for r, s in solns_pell:
                        _a = (r + s * sqrt(D)) * (T + U * sqrt(D)) ** t
                        _b = (r - s * sqrt(D)) * (T - U * sqrt(D)) ** t
                        x_n = _mexpand(S(_a + _b) / 2)
                        y_n = _mexpand(S(_a - _b) / (2 * sqrt(D)))
                        s = P * Matrix([x_n, y_n]) + Q
                        result.add(s)
                else:
                    L = ilcm(*[_.q for _ in P[:4] + Q[:2]])
                    k = 1
                    T_k = T
                    U_k = U
                    while (T_k - 1) % L != 0 or U_k % L != 0:
                        T_k, U_k = (T_k * T + D * U_k * U, T_k * U + U_k * T)
                        k += 1
                    for X, Y in solns_pell:
                        for i in range(k):
                            if all((_is_int(_) for _ in P * Matrix([X, Y]) + Q)):
                                _a = (X + sqrt(D) * Y) * (T_k + sqrt(D) * U_k) ** t
                                _b = (X - sqrt(D) * Y) * (T_k - sqrt(D) * U_k) ** t
                                Xt = S(_a + _b) / 2
                                Yt = S(_a - _b) / (2 * sqrt(D))
                                s = P * Matrix([Xt, Yt]) + Q
                                result.add(s)
                            X, Y = (X * T + D * U * Y, X * U + Y * T)
        return result