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
def _eigenvals_eigenvects_mpmath(M):
    norm2 = lambda v: mp.sqrt(sum((i ** 2 for i in v)))
    v1 = None
    prec = max([x._prec for x in M.atoms(Float)])
    eps = 2 ** (-prec)
    while prec < DEFAULT_MAXPREC:
        with workprec(prec):
            A = mp.matrix(M.evalf(n=prec_to_dps(prec)))
            E, ER = mp.eig(A)
            v2 = norm2([i for e in E for i in (mp.re(e), mp.im(e))])
            if v1 is not None and mp.fabs(v1 - v2) < eps:
                return (E, ER)
            v1 = v2
        prec *= 2
    raise PrecisionExhausted