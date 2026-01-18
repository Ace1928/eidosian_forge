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
def _transformation_to_normal(var, coeff):
    _var = list(var)
    x, y, z = var
    if not any((coeff[i ** 2] for i in var)):
        a = coeff[x * y]
        b = coeff[y * z]
        c = coeff[x * z]
        swap = False
        if not a:
            swap = True
            a, b = (b, a)
        T = Matrix(((1, 1, -b / a), (1, -1, -c / a), (0, 0, 1)))
        if swap:
            T.row_swap(0, 1)
            T.col_swap(0, 1)
        return T
    if coeff[x ** 2] == 0:
        if coeff[y ** 2] == 0:
            _var[0], _var[2] = (var[2], var[0])
            T = _transformation_to_normal(_var, coeff)
            T.row_swap(0, 2)
            T.col_swap(0, 2)
            return T
        else:
            _var[0], _var[1] = (var[1], var[0])
            T = _transformation_to_normal(_var, coeff)
            T.row_swap(0, 1)
            T.col_swap(0, 1)
            return T
    if coeff[x * y] != 0 or coeff[x * z] != 0:
        A = coeff[x ** 2]
        B = coeff[x * y]
        C = coeff[x * z]
        D = coeff[y ** 2]
        E = coeff[y * z]
        F = coeff[z ** 2]
        _coeff = {}
        _coeff[x ** 2] = 4 * A ** 2
        _coeff[y ** 2] = 4 * A * D - B ** 2
        _coeff[z ** 2] = 4 * A * F - C ** 2
        _coeff[y * z] = 4 * A * E - 2 * B * C
        _coeff[x * y] = 0
        _coeff[x * z] = 0
        T_0 = _transformation_to_normal(_var, _coeff)
        return Matrix(3, 3, [1, S(-B) / (2 * A), S(-C) / (2 * A), 0, 1, 0, 0, 0, 1]) * T_0
    elif coeff[y * z] != 0:
        if coeff[y ** 2] == 0:
            if coeff[z ** 2] == 0:
                return Matrix(3, 3, [1, 0, 0, 0, 1, 1, 0, 1, -1])
            else:
                _var[0], _var[2] = (var[2], var[0])
                T = _transformation_to_normal(_var, coeff)
                T.row_swap(0, 2)
                T.col_swap(0, 2)
                return T
        else:
            _var[0], _var[1] = (var[1], var[0])
            T = _transformation_to_normal(_var, coeff)
            T.row_swap(0, 1)
            T.col_swap(0, 1)
            return T
    else:
        return Matrix.eye(3)