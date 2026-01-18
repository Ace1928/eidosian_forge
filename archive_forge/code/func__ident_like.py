from warnings import warn
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from scipy.linalg._decomp_qr import qr
from scipy.sparse._sputils import is_pydata_spmatrix
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg._interface import IdentityOperator
from scipy.sparse.linalg._onenormest import onenormest
def _ident_like(A):
    if scipy.sparse.issparse(A):
        out = scipy.sparse.eye(A.shape[0], A.shape[1], dtype=A.dtype)
        if isinstance(A, scipy.sparse.spmatrix):
            return out.asformat(A.format)
        return scipy.sparse.dia_array(out).asformat(A.format)
    elif is_pydata_spmatrix(A):
        import sparse
        return sparse.eye(A.shape[0], A.shape[1], dtype=A.dtype)
    elif isinstance(A, scipy.sparse.linalg.LinearOperator):
        return IdentityOperator(A.shape, dtype=A.dtype)
    else:
        return np.eye(A.shape[0], A.shape[1], dtype=A.dtype)