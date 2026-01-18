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
def _eigenspace(M, eigenval, iszerofunc=_iszero, simplify=False):
    """Get a basis for the eigenspace for a particular eigenvalue"""
    m = M - M.eye(M.rows) * eigenval
    ret = m.nullspace(iszerofunc=iszerofunc)
    if len(ret) == 0 and simplify:
        ret = m.nullspace(iszerofunc=iszerofunc, simplify=True)
    if len(ret) == 0:
        raise NotImplementedError("Can't evaluate eigenvector for eigenvalue {}".format(eigenval))
    return ret