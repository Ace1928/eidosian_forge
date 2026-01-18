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
class IterOpInv(LinearOperator):
    """
    IterOpInv:
       helper class to repeatedly solve [A-sigma*M]*x = b
       using an iterative method
    """

    def __init__(self, A, M, sigma, ifunc=gmres_loose, tol=0):
        self.A = A
        self.M = M
        self.sigma = sigma

        def mult_func(x):
            return A.matvec(x) - sigma * M.matvec(x)

        def mult_func_M_None(x):
            return A.matvec(x) - sigma * x
        x = np.zeros(A.shape[1])
        if M is None:
            dtype = mult_func_M_None(x).dtype
            self.OP = LinearOperator(self.A.shape, mult_func_M_None, dtype=dtype)
        else:
            dtype = mult_func(x).dtype
            self.OP = LinearOperator(self.A.shape, mult_func, dtype=dtype)
        self.shape = A.shape
        if tol <= 0:
            tol = 2 * np.finfo(self.OP.dtype).eps
        self.ifunc = ifunc
        self.tol = tol

    def _matvec(self, x):
        b, info = self.ifunc(self.OP, x, tol=self.tol)
        if info != 0:
            raise ValueError('Error in inverting [A-sigma*M]: function %s did not converge (info = %i).' % (self.ifunc.__name__, info))
        return b

    @property
    def dtype(self):
        return self.OP.dtype