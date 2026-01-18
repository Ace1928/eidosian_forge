from sympy.core.numbers import mod_inverse
from .common import MatrixError, NonSquareMatrixError, NonInvertibleMatrixError
from .utilities import _iszero
def _inv_GE(M, iszerofunc=_iszero):
    """Calculates the inverse using Gaussian elimination.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_LU
    inverse_CH
    inverse_LDL
    """
    from .dense import Matrix
    if not M.is_square:
        raise NonSquareMatrixError('A Matrix must be square to invert.')
    big = Matrix.hstack(M.as_mutable(), Matrix.eye(M.rows))
    red = big.rref(iszerofunc=iszerofunc, simplify=True)[0]
    if any((iszerofunc(red[j, j]) for j in range(red.rows))):
        raise NonInvertibleMatrixError('Matrix det == 0; not invertible.')
    return M._new(red[:, big.rows:])