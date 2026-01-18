from sympy.core.function import expand_mul
from sympy.core.symbol import Dummy, uniquely_named_symbol, symbols
from sympy.utilities.iterables import numbered_symbols
from .common import ShapeError, NonSquareMatrixError, NonInvertibleMatrixError
from .eigen import _fuzzy_positive_definite
from .utilities import _get_intermediate_simp, _iszero
def _lower_triangular_solve_sparse(M, rhs):
    """Solves ``Ax = B``, where A is a lower triangular matrix.

    See Also
    ========

    upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    """
    if not M.is_square:
        raise NonSquareMatrixError('Matrix must be square.')
    if rhs.rows != M.rows:
        raise ShapeError('Matrices size mismatch.')
    if not M.is_lower:
        raise ValueError('Matrix must be lower triangular.')
    dps = _get_intermediate_simp()
    rows = [[] for i in range(M.rows)]
    for i, j, v in M.row_list():
        if i > j:
            rows[i].append((j, v))
    X = rhs.as_mutable()
    for j in range(rhs.cols):
        for i in range(rhs.rows):
            for u, v in rows[i]:
                X[i, j] -= v * X[u, j]
            X[i, j] = dps(X[i, j] / M[i, i])
    return M._new(X)