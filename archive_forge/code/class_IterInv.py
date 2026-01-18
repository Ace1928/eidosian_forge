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
class IterInv(LinearOperator):
    """
    IterInv:
       helper class to repeatedly solve M*x=b
       using an iterative method.
    """

    def __init__(self, M, ifunc=gmres_loose, tol=0):
        self.M = M
        if hasattr(M, 'dtype'):
            self.dtype = M.dtype
        else:
            x = np.zeros(M.shape[1])
            self.dtype = (M * x).dtype
        self.shape = M.shape
        if tol <= 0:
            tol = 2 * np.finfo(self.dtype).eps
        self.ifunc = ifunc
        self.tol = tol

    def _matvec(self, x):
        b, info = self.ifunc(self.M, x, tol=self.tol)
        if info != 0:
            raise ValueError('Error in inverting M: function %s did not converge (info = %i).' % (self.ifunc.__name__, info))
        return b