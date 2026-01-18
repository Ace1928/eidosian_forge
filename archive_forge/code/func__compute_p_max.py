from warnings import warn
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from scipy.linalg._decomp_qr import qr
from scipy.sparse._sputils import is_pydata_spmatrix
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg._interface import IdentityOperator
from scipy.sparse.linalg._onenormest import onenormest
def _compute_p_max(m_max):
    """
    Compute the largest positive integer p such that p*(p-1) <= m_max + 1.

    Do this in a slightly dumb way, but safe and not too slow.

    Parameters
    ----------
    m_max : int
        A count related to bounds.

    """
    sqrt_m_max = np.sqrt(m_max)
    p_low = int(np.floor(sqrt_m_max))
    p_high = int(np.ceil(sqrt_m_max + 1))
    return max((p for p in range(p_low, p_high + 1) if p * (p - 1) <= m_max + 1))