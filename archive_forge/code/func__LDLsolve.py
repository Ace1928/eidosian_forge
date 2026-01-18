from sympy.core.function import expand_mul
from sympy.core.symbol import Dummy, uniquely_named_symbol, symbols
from sympy.utilities.iterables import numbered_symbols
from .common import ShapeError, NonSquareMatrixError, NonInvertibleMatrixError
from .eigen import _fuzzy_positive_definite
from .utilities import _get_intermediate_simp, _iszero
def _LDLsolve(M, rhs):
    """Solves ``Ax = B`` using LDL decomposition,
    for a general square and non-singular matrix.

    For a non-square matrix with rows > cols,
    the least squares solution is returned.

    Examples
    ========

    >>> from sympy import Matrix, eye
    >>> A = eye(2)*2
    >>> B = Matrix([[1, 2], [3, 4]])
    >>> A.LDLsolve(B) == B/2
    True

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.LDLdecomposition
    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LUsolve
    QRsolve
    pinv_solve
    """
    if M.rows < M.cols:
        raise NotImplementedError('Under-determined System. Try M.gauss_jordan_solve(rhs)')
    hermitian = True
    reform = False
    if M.is_symmetric():
        hermitian = False
    elif not M.is_hermitian:
        reform = True
    if reform or _fuzzy_positive_definite(M) is False:
        H = M.H
        M = H.multiply(M)
        rhs = H.multiply(rhs)
        hermitian = not M.is_symmetric()
    L, D = M.LDLdecomposition(hermitian=hermitian)
    Y = L.lower_triangular_solve(rhs)
    Z = D.diagonal_solve(Y)
    if hermitian:
        return L.H.upper_triangular_solve(Z)
    else:
        return L.T.upper_triangular_solve(Z)