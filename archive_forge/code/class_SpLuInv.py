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
class SpLuInv(LinearOperator):
    """
    SpLuInv:
       helper class to repeatedly solve M*x=b
       using a sparse LU-decomposition of M
    """

    def __init__(self, M):
        self.M_lu = splu(M)
        self.shape = M.shape
        self.dtype = M.dtype
        self.isreal = not np.issubdtype(self.dtype, np.complexfloating)

    def _matvec(self, x):
        x = np.asarray(x)
        if self.isreal and np.issubdtype(x.dtype, np.complexfloating):
            return self.M_lu.solve(np.real(x).astype(self.dtype)) + 1j * self.M_lu.solve(np.imag(x).astype(self.dtype))
        else:
            return self.M_lu.solve(x.astype(self.dtype))