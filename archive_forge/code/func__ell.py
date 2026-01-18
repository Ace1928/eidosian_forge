import numpy as np
from scipy.linalg._basic import solve, solve_triangular
from scipy.sparse._base import issparse
from scipy.sparse.linalg import spsolve
from scipy.sparse._sputils import is_pydata_spmatrix, isintlike
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse._construct import eye
from ._expm_multiply import _ident_like, _exact_1_norm as _onenorm
def _ell(A, m):
    """
    A helper function for expm_2009.

    Parameters
    ----------
    A : linear operator
        A linear operator whose norm of power we care about.
    m : int
        The power of the linear operator

    Returns
    -------
    value : int
        A value related to a bound.

    """
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    c_i = {3: 100800.0, 5: 10059033600.0, 7: 4487938430976000.0, 9: 5.914384781877412e+21, 13: 1.1325077560602111e+35}
    abs_c_recip = c_i[m]
    u = 2 ** (-53)
    A_abs_onenorm = _onenorm_matrix_power_nnm(abs(A), 2 * m + 1)
    if not A_abs_onenorm:
        return 0
    alpha = A_abs_onenorm / (_onenorm(A) * abs_c_recip)
    log2_alpha_div_u = np.log2(alpha / u)
    value = int(np.ceil(log2_alpha_div_u / (2 * m)))
    return max(value, 0)