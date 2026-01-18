import copy
from sympy.core import S
from sympy.core.function import expand_mul
from sympy.functions.elementary.miscellaneous import Min, sqrt
from sympy.functions.elementary.complexes import sign
from .common import NonSquareMatrixError, NonPositiveDefiniteMatrixError
from .utilities import _get_intermediate_simp, _iszero
from .determinant import _find_reasonable_pivot_naive
def _upper_hessenberg_decomposition(A):
    """Converts a matrix into Hessenberg matrix H.

    Returns 2 matrices H, P s.t.
    $P H P^{T} = A$, where H is an upper hessenberg matrix
    and P is an orthogonal matrix

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([
    ...     [1,2,3],
    ...     [-3,5,6],
    ...     [4,-8,9],
    ... ])
    >>> H, P = A.upper_hessenberg_decomposition()
    >>> H
    Matrix([
    [1,    6/5,    17/5],
    [5, 213/25, -134/25],
    [0, 216/25,  137/25]])
    >>> P
    Matrix([
    [1,    0,   0],
    [0, -3/5, 4/5],
    [0,  4/5, 3/5]])
    >>> P * H * P.H == A
    True


    References
    ==========

    .. [#] https://mathworld.wolfram.com/HessenbergDecomposition.html
    """
    M = A.as_mutable()
    if not M.is_square:
        raise NonSquareMatrixError('Matrix must be square.')
    n = M.cols
    P = M.eye(n)
    H = M
    for j in range(n - 2):
        u = H[j + 1:, j]
        if u[1:, :].is_zero_matrix:
            continue
        if sign(u[0]) != 0:
            u[0] = u[0] + sign(u[0]) * u.norm()
        else:
            u[0] = u[0] + u.norm()
        v = u / u.norm()
        H[j + 1:, :] = H[j + 1:, :] - 2 * v * (v.H * H[j + 1:, :])
        H[:, j + 1:] = H[:, j + 1:] - H[:, j + 1:] * (2 * v) * v.H
        P[:, j + 1:] = P[:, j + 1:] - P[:, j + 1:] * (2 * v) * v.H
    return (H, P)