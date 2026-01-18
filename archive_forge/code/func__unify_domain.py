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
@classmethod
def _unify_domain(cls, *matrices):
    """Convert matrices to a common domain"""
    domains = {matrix.domain for matrix in matrices}
    if len(domains) == 1:
        return matrices
    domain = reduce(lambda x, y: x.unify(y), domains)
    return tuple((matrix.convert_to(domain) for matrix in matrices))