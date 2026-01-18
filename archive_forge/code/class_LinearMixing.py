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
class LinearMixing(GenericBroyden):
    """
    Find a root of a function, using a scalar Jacobian approximation.

    .. warning::

       This algorithm may be useful for specific problems, but whether
       it will work may depend strongly on the problem.

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        The Jacobian approximation is (-1/alpha).
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='linearmixing'`` in particular.

    """

    def __init__(self, alpha=None):
        GenericBroyden.__init__(self)
        self.alpha = alpha

    def solve(self, f, tol=0):
        return -f * self.alpha

    def matvec(self, f):
        return -f / self.alpha

    def rsolve(self, f, tol=0):
        return -f * np.conj(self.alpha)

    def rmatvec(self, f):
        return -f / np.conj(self.alpha)

    def todense(self):
        return np.diag(np.full(self.shape[0], -1 / self.alpha))

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        pass