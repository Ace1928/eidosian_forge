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
def _is_diagonalizable_with_eigen(M, reals_only=False):
    """See _is_diagonalizable. This function returns the bool along with the
    eigenvectors to avoid calculating them again in functions like
    ``diagonalize``."""
    if not M.is_square:
        return (False, [])
    eigenvecs = M.eigenvects(simplify=True)
    for val, mult, basis in eigenvecs:
        if reals_only and (not val.is_real):
            return (False, eigenvecs)
        if mult != len(basis):
            return (False, eigenvecs)
    return (True, eigenvecs)