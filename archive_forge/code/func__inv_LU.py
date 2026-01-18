from sympy.core.numbers import mod_inverse
from .common import MatrixError, NonSquareMatrixError, NonInvertibleMatrixError
from .utilities import _iszero
def _inv_LU(M, iszerofunc=_iszero):
    """Calculates the inverse using LU decomposition.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_CH
    inverse_LDL
    """
    if not M.is_square:
        raise NonSquareMatrixError('A Matrix must be square to invert.')
    if M.free_symbols:
        _verify_invertible(M, iszerofunc=iszerofunc)
    return M.LUsolve(M.eye(M.rows), iszerofunc=_iszero)