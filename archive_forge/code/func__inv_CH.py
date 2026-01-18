from sympy.core.numbers import mod_inverse
from .common import MatrixError, NonSquareMatrixError, NonInvertibleMatrixError
from .utilities import _iszero
def _inv_CH(M, iszerofunc=_iszero):
    """Calculates the inverse using cholesky decomposition.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_LU
    inverse_LDL
    """
    _verify_invertible(M, iszerofunc=iszerofunc)
    return M.cholesky_solve(M.eye(M.rows))