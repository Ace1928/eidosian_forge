from types import FunctionType
from sympy.core.numbers import Float, Integer
from sympy.core.singleton import S
from sympy.core.symbol import uniquely_named_symbol
from sympy.core.mul import Mul
from sympy.polys import PurePoly, cancel
from sympy.functions.combinatorial.numbers import nC
from sympy.polys.matrices.domainmatrix import DomainMatrix
from .common import NonSquareMatrixError
from .utilities import (
def _cofactor(M, i, j, method='berkowitz'):
    """Calculate the cofactor of an element.

    Parameters
    ==========

    method : string, optional
        Method to use to find the cofactors, can be "bareiss", "berkowitz" or
        "lu".

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> M.cofactor(0, 1)
    -3

    See Also
    ========

    cofactor_matrix
    minor
    minor_submatrix
    """
    if not M.is_square or M.rows < 1:
        raise NonSquareMatrixError()
    return S.NegativeOne ** ((i + j) % 2) * M.minor(i, j, method)