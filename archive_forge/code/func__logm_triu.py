import warnings
import numpy as np
from scipy.linalg._matfuncs_sqrtm import SqrtmError, _sqrtm_triu
from scipy.linalg._decomp_schur import schur, rsf2csf
from scipy.linalg._matfuncs import funm
from scipy.linalg import svdvals, solve_triangular
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg import onenormest
import scipy.special
def _logm_triu(T):
    """
    Compute matrix logarithm of an upper triangular matrix.

    The matrix logarithm is the inverse of
    expm: expm(logm(`T`)) == `T`

    Parameters
    ----------
    T : (N, N) array_like
        Upper triangular matrix whose logarithm to evaluate

    Returns
    -------
    logm : (N, N) ndarray
        Matrix logarithm of `T`

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2012)
           "Improved Inverse Scaling and Squaring Algorithms
           for the Matrix Logarithm."
           SIAM Journal on Scientific Computing, 34 (4). C152-C169.
           ISSN 1095-7197

    .. [2] Nicholas J. Higham (2008)
           "Functions of Matrices: Theory and Computation"
           ISBN 978-0-898716-46-7

    .. [3] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    """
    T = np.asarray(T)
    if len(T.shape) != 2 or T.shape[0] != T.shape[1]:
        raise ValueError('expected an upper triangular square matrix')
    n, n = T.shape
    T_diag = np.diag(T)
    keep_it_real = np.isrealobj(T) and np.min(T_diag) >= 0
    if keep_it_real:
        T0 = T
    else:
        T0 = T.astype(complex)
    theta = (None, 1.59e-05, 0.00231, 0.0194, 0.0621, 0.128, 0.206, 0.288, 0.367, 0.439, 0.503, 0.56, 0.609, 0.652, 0.689, 0.721, 0.749)
    R, s, m = _inverse_squaring_helper(T0, theta)
    nodes, weights = scipy.special.p_roots(m)
    nodes = nodes.real
    if nodes.shape != (m,) or weights.shape != (m,):
        raise Exception('internal error')
    nodes = 0.5 + 0.5 * nodes
    weights = 0.5 * weights
    ident = np.identity(n)
    U = np.zeros_like(R)
    for alpha, beta in zip(weights, nodes):
        U += solve_triangular(ident + beta * R, alpha * R)
    U *= np.exp2(s)
    has_principal_branch = all((x.real > 0 or x.imag != 0 for x in np.diag(T0)))
    if has_principal_branch:
        U[np.diag_indices(n)] = np.log(np.diag(T0))
        for i in range(n - 1):
            l1 = T0[i, i]
            l2 = T0[i + 1, i + 1]
            t12 = T0[i, i + 1]
            U[i, i + 1] = _logm_superdiag_entry(l1, l2, t12)
    if not np.array_equal(U, np.triu(U)):
        raise Exception('U is not upper triangular')
    return U