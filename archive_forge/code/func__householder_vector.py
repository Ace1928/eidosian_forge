from types import FunctionType
from collections import Counter
from mpmath import mp, workprec
from mpmath.libmp.libmpf import prec_to_dps
from sympy.core.sorting import default_sort_key
from sympy.core.evalf import DEFAULT_MAXPREC, PrecisionExhausted
from sympy.core.logic import fuzzy_and, fuzzy_or
from sympy.core.numbers import Float
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys import roots, CRootOf, ZZ, QQ, EX
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.eigen import dom_eigenvects, dom_eigenvects_to_sympy
from sympy.polys.polytools import gcd
from .common import MatrixError, NonSquareMatrixError
from .determinant import _find_reasonable_pivot
from .utilities import _iszero, _simplify
def _householder_vector(x):
    if not x.cols == 1:
        raise ValueError('Input must be a column matrix')
    v = x.copy()
    v_plus = x.copy()
    v_minus = x.copy()
    q = x[0, 0] / abs(x[0, 0])
    norm_x = x.norm()
    v_plus[0, 0] = x[0, 0] + q * norm_x
    v_minus[0, 0] = x[0, 0] - q * norm_x
    if x[1:, 0].norm() == 0:
        bet = 0
        v[0, 0] = 1
    else:
        if v_plus.norm() <= v_minus.norm():
            v = v_plus
        else:
            v = v_minus
        v = v / v[0]
        bet = 2 / v.norm() ** 2
    return (v, bet)