from sympy.core.numbers import mod_inverse
from .common import MatrixError, NonSquareMatrixError, NonInvertibleMatrixError
from .utilities import _iszero
def _pinv_full_rank(M):
    """Subroutine for full row or column rank matrices.

    For full row rank matrices, inverse of ``A * A.H`` Exists.
    For full column rank matrices, inverse of ``A.H * A`` Exists.

    This routine can apply for both cases by checking the shape
    and have small decision.
    """
    if M.is_zero_matrix:
        return M.H
    if M.rows >= M.cols:
        return M.H.multiply(M).inv().multiply(M.H)
    else:
        return M.H.multiply(M.multiply(M.H).inv())