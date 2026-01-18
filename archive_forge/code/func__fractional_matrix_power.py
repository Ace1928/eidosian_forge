import warnings
import numpy as np
from scipy.linalg._matfuncs_sqrtm import SqrtmError, _sqrtm_triu
from scipy.linalg._decomp_schur import schur, rsf2csf
from scipy.linalg._matfuncs import funm
from scipy.linalg import svdvals, solve_triangular
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg import onenormest
import scipy.special
def _fractional_matrix_power(A, p):
    """
    Compute the fractional power of a matrix.

    See the fractional_matrix_power docstring in matfuncs.py for more info.

    """
    A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')
    if p == int(p):
        return np.linalg.matrix_power(A, int(p))
    s = svdvals(A)
    if s[-1]:
        k2 = s[0] / s[-1]
        p1 = p - np.floor(p)
        p2 = p - np.ceil(p)
        if p1 * k2 ** (1 - p1) <= -p2 * k2:
            a = int(np.floor(p))
            b = p1
        else:
            a = int(np.ceil(p))
            b = p2
        try:
            R = _remainder_matrix_power(A, b)
            Q = np.linalg.matrix_power(A, a)
            return Q.dot(R)
        except np.linalg.LinAlgError:
            pass
    if p < 0:
        X = np.empty_like(A)
        X.fill(np.nan)
        return X
    else:
        p1 = p - np.floor(p)
        a = int(np.floor(p))
        b = p1
        R, info = funm(A, lambda x: pow(x, b), disp=False)
        Q = np.linalg.matrix_power(A, a)
        return Q.dot(R)