from warnings import warn
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from scipy.linalg._decomp_qr import qr
from scipy.sparse._sputils import is_pydata_spmatrix
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg._interface import IdentityOperator
from scipy.sparse.linalg._onenormest import onenormest
def _expm_multiply_interval_core_0(A, X, h, mu, q, norm_info, tol, ell, n0):
    """
    A helper function, for the case q <= s.
    """
    if norm_info.onenorm() == 0:
        m_star, s = (0, 1)
    else:
        norm_info.set_scale(1.0 / q)
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)
        norm_info.set_scale(1)
    for k in range(q):
        X[k + 1] = _expm_multiply_simple_core(A, X[k], h, mu, m_star, s)
    return (X, 0)