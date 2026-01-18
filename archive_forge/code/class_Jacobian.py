import sys
import numpy as np
from scipy.linalg import norm, solve, inv, qr, svd, LinAlgError
from numpy import asarray, dot, vdot
import scipy.sparse.linalg
import scipy.sparse
from scipy.linalg import get_blas_funcs
import inspect
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from ._linesearch import scalar_search_wolfe1, scalar_search_armijo
class Jacobian:
    """
    Common interface for Jacobians or Jacobian approximations.

    The optional methods come useful when implementing trust region
    etc., algorithms that often require evaluating transposes of the
    Jacobian.

    Methods
    -------
    solve
        Returns J^-1 * v
    update
        Updates Jacobian to point `x` (where the function has residual `Fx`)

    matvec : optional
        Returns J * v
    rmatvec : optional
        Returns A^H * v
    rsolve : optional
        Returns A^-H * v
    matmat : optional
        Returns A * V, where V is a dense matrix with dimensions (N,K).
    todense : optional
        Form the dense Jacobian matrix. Necessary for dense trust region
        algorithms, and useful for testing.

    Attributes
    ----------
    shape
        Matrix dimensions (M, N)
    dtype
        Data type of the matrix.
    func : callable, optional
        Function the Jacobian corresponds to

    """

    def __init__(self, **kw):
        names = ['solve', 'update', 'matvec', 'rmatvec', 'rsolve', 'matmat', 'todense', 'shape', 'dtype']
        for name, value in kw.items():
            if name not in names:
                raise ValueError('Unknown keyword argument %s' % name)
            if value is not None:
                setattr(self, name, kw[name])
        if hasattr(self, 'todense'):
            self.__array__ = lambda: self.todense()

    def aspreconditioner(self):
        return InverseJacobian(self)

    def solve(self, v, tol=0):
        raise NotImplementedError

    def update(self, x, F):
        pass

    def setup(self, x, F, func):
        self.func = func
        self.shape = (F.size, x.size)
        self.dtype = F.dtype
        if self.__class__.setup is Jacobian.setup:
            self.update(x, F)