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
def _transformation_to_DN(var, coeff):
    x, y = var
    a = coeff[x ** 2]
    b = coeff[x * y]
    c = coeff[y ** 2]
    d = coeff[x]
    e = coeff[y]
    f = coeff[1]
    a, b, c, d, e, f = [as_int(i) for i in _remove_gcd(a, b, c, d, e, f)]
    X, Y = symbols('X, Y', integer=True)
    if b:
        B, C = _rational_pq(2 * a, b)
        A, T = _rational_pq(a, B ** 2)
        coeff = {X ** 2: A * B, X * Y: 0, Y ** 2: B * (c * T - A * C ** 2), X: d * T, Y: B * e * T - d * T * C, 1: f * T * B}
        A_0, B_0 = _transformation_to_DN([X, Y], coeff)
        return (Matrix(2, 2, [S.One / B, -S(C) / B, 0, 1]) * A_0, Matrix(2, 2, [S.One / B, -S(C) / B, 0, 1]) * B_0)
    elif d:
        B, C = _rational_pq(2 * a, d)
        A, T = _rational_pq(a, B ** 2)
        coeff = {X ** 2: A, X * Y: 0, Y ** 2: c * T, X: 0, Y: e * T, 1: f * T - A * C ** 2}
        A_0, B_0 = _transformation_to_DN([X, Y], coeff)
        return (Matrix(2, 2, [S.One / B, 0, 0, 1]) * A_0, Matrix(2, 2, [S.One / B, 0, 0, 1]) * B_0 + Matrix([-S(C) / B, 0]))
    elif e:
        B, C = _rational_pq(2 * c, e)
        A, T = _rational_pq(c, B ** 2)
        coeff = {X ** 2: a * T, X * Y: 0, Y ** 2: A, X: 0, Y: 0, 1: f * T - A * C ** 2}
        A_0, B_0 = _transformation_to_DN([X, Y], coeff)
        return (Matrix(2, 2, [1, 0, 0, S.One / B]) * A_0, Matrix(2, 2, [1, 0, 0, S.One / B]) * B_0 + Matrix([0, -S(C) / B]))
    else:
        return (Matrix(2, 2, [S.One / a, 0, 0, 1]), Matrix([0, 0]))