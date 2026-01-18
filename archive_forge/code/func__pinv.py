from sympy.core.numbers import mod_inverse
from .common import MatrixError, NonSquareMatrixError, NonInvertibleMatrixError
from .utilities import _iszero
def _pinv(M, method='RD'):
    """Calculate the Moore-Penrose pseudoinverse of the matrix.

    The Moore-Penrose pseudoinverse exists and is unique for any matrix.
    If the matrix is invertible, the pseudoinverse is the same as the
    inverse.

    Parameters
    ==========

    method : String, optional
        Specifies the method for computing the pseudoinverse.

        If ``'RD'``, Rank-Decomposition will be used.

        If ``'ED'``, Diagonalization will be used.

    Examples
    ========

    Computing pseudoinverse by rank decomposition :

    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
    >>> A.pinv()
    Matrix([
    [-17/18,  4/9],
    [  -1/9,  1/9],
    [ 13/18, -2/9]])

    Computing pseudoinverse by diagonalization :

    >>> B = A.pinv(method='ED')
    >>> B.simplify()
    >>> B
    Matrix([
    [-17/18,  4/9],
    [  -1/9,  1/9],
    [ 13/18, -2/9]])

    See Also
    ========

    inv
    pinv_solve

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Moore-Penrose_pseudoinverse

    """
    if M.is_zero_matrix:
        return M.H
    if method == 'RD':
        return _pinv_rank_decomposition(M)
    elif method == 'ED':
        return _pinv_diagonalization(M)
    else:
        raise ValueError('invalid pinv method %s' % repr(method))