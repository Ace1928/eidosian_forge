from sympy.core.function import expand_mul
from sympy.core.symbol import Dummy, uniquely_named_symbol, symbols
from sympy.utilities.iterables import numbered_symbols
from .common import ShapeError, NonSquareMatrixError, NonInvertibleMatrixError
from .eigen import _fuzzy_positive_definite
from .utilities import _get_intermediate_simp, _iszero
def _diagonal_solve(M, rhs):
    """Solves ``Ax = B`` efficiently, where A is a diagonal Matrix,
    with non-zero diagonal entries.

    Examples
    ========

    >>> from sympy import Matrix, eye
    >>> A = eye(2)*2
    >>> B = Matrix([[1, 2], [3, 4]])
    >>> A.diagonal_solve(B) == B/2
    True

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    """
    if not M.is_diagonal():
        raise TypeError('Matrix should be diagonal')
    if rhs.rows != M.rows:
        raise TypeError('Size mismatch')
    return M._new(rhs.rows, rhs.cols, lambda i, j: rhs[i, j] / M[i, i])