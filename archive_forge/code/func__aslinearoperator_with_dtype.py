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
def _aslinearoperator_with_dtype(m):
    m = aslinearoperator(m)
    if not hasattr(m, 'dtype'):
        x = np.zeros(m.shape[1])
        m.dtype = (m * x).dtype
    return m