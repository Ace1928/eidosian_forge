import warnings
import numpy as np
from scipy.linalg._matfuncs_sqrtm import SqrtmError, _sqrtm_triu
from scipy.linalg._decomp_schur import schur, rsf2csf
from scipy.linalg._matfuncs import funm
from scipy.linalg import svdvals, solve_triangular
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg import onenormest
import scipy.special
def _fractional_power_pade(R, t, m):
    """
    Evaluate the Pade approximation of a fractional matrix power.

    Evaluate the degree-m Pade approximation of R
    to the fractional matrix power t using the continued fraction
    in bottom-up fashion using algorithm (4.1) in [1]_.

    Parameters
    ----------
    R : (N, N) array_like
        Upper triangular matrix whose fractional power to evaluate.
    t : float
        Fractional power between -1 and 1 exclusive.
    m : positive integer
        Degree of Pade approximation.

    Returns
    -------
    U : (N, N) array_like
        The degree-m Pade approximation of R to the fractional power t.
        This matrix will be upper triangular.

    References
    ----------
    .. [1] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    """
    if m < 1 or int(m) != m:
        raise ValueError('expected a positive integer m')
    if not -1 < t < 1:
        raise ValueError('expected -1 < t < 1')
    R = np.asarray(R)
    if len(R.shape) != 2 or R.shape[0] != R.shape[1]:
        raise ValueError('expected an upper triangular square matrix')
    n, n = R.shape
    ident = np.identity(n)
    Y = R * _fractional_power_pade_constant(2 * m, t)
    for j in range(2 * m - 1, 0, -1):
        rhs = R * _fractional_power_pade_constant(j, t)
        Y = solve_triangular(ident + Y, rhs)
    U = ident + Y
    if not np.array_equal(U, np.triu(U)):
        raise Exception('U is not upper triangular')
    return U