import numpy as np
import warnings
from scipy.sparse.linalg._interface import aslinearoperator, LinearOperator
from scipy.sparse import eye, issparse
from scipy.linalg import eig, eigh, lu_factor, lu_solve
from scipy.sparse._sputils import isdense, is_pydata_spmatrix
from scipy.sparse.linalg import gmres, splu
from scipy._lib._util import _aligned_zeros
from scipy._lib._threadsafety import ReentrancyLock
from . import _arpack
def _fast_spmatrix_to_csc(A, hermitian=False):
    """Convert sparse matrix to CSC (by transposing, if possible)"""
    if A.format == 'csr' and hermitian and (not np.issubdtype(A.dtype, np.complexfloating)):
        return A.T
    elif is_pydata_spmatrix(A):
        return A
    else:
        return A.tocsc()