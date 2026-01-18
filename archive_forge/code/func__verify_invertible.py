from sympy.core.numbers import mod_inverse
from .common import MatrixError, NonSquareMatrixError, NonInvertibleMatrixError
from .utilities import _iszero
def _verify_invertible(M, iszerofunc=_iszero):
    """Initial check to see if a matrix is invertible. Raises or returns
    determinant for use in _inv_ADJ."""
    if not M.is_square:
        raise NonSquareMatrixError('A Matrix must be square to invert.')
    d = M.det(method='berkowitz')
    zero = d.equals(0)
    if zero is None:
        ok = M.rref(simplify=True)[0]
        zero = any((iszerofunc(ok[j, j]) for j in range(ok.rows)))
    if zero:
        raise NonInvertibleMatrixError('Matrix det == 0; not invertible.')
    return d