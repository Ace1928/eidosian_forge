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
def _raise_no_convergence(self):
    msg = 'No convergence (%d iterations, %d/%d eigenvectors converged)'
    k_ok = self.iparam[4]
    num_iter = self.iparam[2]
    try:
        ev, vec = self.extract(True)
    except ArpackError as err:
        msg = f'{msg} [{err}]'
        ev = np.zeros((0,))
        vec = np.zeros((self.n, 0))
        k_ok = 0
    raise ArpackNoConvergence(msg % (num_iter, k_ok, self.k), ev, vec)