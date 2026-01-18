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
def _is_indefinite(M):
    if M.is_hermitian:
        eigen = M.eigenvals()
        args1 = [x.is_positive for x in eigen.keys()]
        any_positive = fuzzy_or(args1)
        args2 = [x.is_negative for x in eigen.keys()]
        any_negative = fuzzy_or(args2)
        return fuzzy_and([any_positive, any_negative])
    elif M.is_square:
        return (M + M.H).is_indefinite
    return False