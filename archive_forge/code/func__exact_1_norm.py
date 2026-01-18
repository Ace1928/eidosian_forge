from warnings import warn
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from scipy.linalg._decomp_qr import qr
from scipy.sparse._sputils import is_pydata_spmatrix
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg._interface import IdentityOperator
from scipy.sparse.linalg._onenormest import onenormest
def _exact_1_norm(A):
    if scipy.sparse.issparse(A):
        return max(abs(A).sum(axis=0).flat)
    elif is_pydata_spmatrix(A):
        return max(abs(A).sum(axis=0))
    else:
        return np.linalg.norm(A, 1)