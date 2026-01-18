from sympy.core.function import expand_mul
from sympy.core.symbol import Dummy, uniquely_named_symbol, symbols
from sympy.utilities.iterables import numbered_symbols
from .common import ShapeError, NonSquareMatrixError, NonInvertibleMatrixError
from .eigen import _fuzzy_positive_definite
from .utilities import _get_intermediate_simp, _iszero
def _upper_triangular_solve(M, rhs):
    """Solves ``Ax = B``, where A is an upper triangular matrix.

    See Also
    ========

    lower_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    """
    from .dense import MutableDenseMatrix
    if not M.is_square:
        raise NonSquareMatrixError('Matrix must be square.')
    if rhs.rows != M.rows:
        raise ShapeError('Matrix size mismatch.')
    if not M.is_upper:
        raise TypeError('Matrix is not upper triangular.')
    dps = _get_intermediate_simp()
    X = MutableDenseMatrix.zeros(M.rows, rhs.cols)
    for j in range(rhs.cols):
        for i in reversed(range(M.rows)):
            if M[i, i] == 0:
                raise ValueError('Matrix must be non-singular.')
            X[i, j] = dps((rhs[i, j] - sum((M[i, k] * X[k, j] for k in range(i + 1, M.rows)))) / M[i, i])
    return M._new(X)