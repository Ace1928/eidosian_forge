from warnings import warn
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from scipy.linalg._decomp_qr import qr
from scipy.sparse._sputils import is_pydata_spmatrix
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg._interface import IdentityOperator
from scipy.sparse.linalg._onenormest import onenormest
def _expm_multiply_interval_core_2(A, X, h, mu, m_star, s, q, tol):
    """
    A helper function, for the case q > s and q % s > 0.
    """
    d = q // s
    j = q // d
    r = q - d * j
    input_shape = X.shape[1:]
    K_shape = (m_star + 1,) + input_shape
    K = np.empty(K_shape, dtype=X.dtype)
    for i in range(j + 1):
        Z = X[i * d]
        K[0] = Z
        high_p = 0
        if i < j:
            effective_d = d
        else:
            effective_d = r
        for k in range(1, effective_d + 1):
            F = K[0]
            c1 = _exact_inf_norm(F)
            for p in range(1, m_star + 1):
                if p == high_p + 1:
                    K[p] = h * A.dot(K[p - 1]) / float(p)
                    high_p = p
                coeff = float(pow(k, p))
                F = F + coeff * K[p]
                inf_norm_K_p_1 = _exact_inf_norm(K[p])
                c2 = coeff * inf_norm_K_p_1
                if c1 + c2 <= tol * _exact_inf_norm(F):
                    break
                c1 = c2
            X[k + i * d] = np.exp(k * h * mu) * F
    return (X, 2)