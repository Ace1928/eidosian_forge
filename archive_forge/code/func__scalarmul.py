from functools import reduce
from typing import Union as tUnion, Tuple as tTuple
from sympy.core.sympify import _sympify
from ..domains import Domain
from ..constructor import construct_domain
from .exceptions import (DMNonSquareMatrixError, DMShapeError,
from .ddm import DDM
from .sdm import SDM
from .domainscalar import DomainScalar
from sympy.polys.domains import ZZ, EXRAW, QQ
def _scalarmul(A, lamda, reverse):
    if lamda == A.domain.zero:
        return DomainMatrix.zeros(A.shape, A.domain)
    elif lamda == A.domain.one:
        return A.copy()
    elif reverse:
        return A.rmul(lamda)
    else:
        return A.mul(lamda)