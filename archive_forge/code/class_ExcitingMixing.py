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
class ExcitingMixing(GenericBroyden):
    """
    Find a root of a function, using a tuned diagonal Jacobian approximation.

    The Jacobian matrix is diagonal and is tuned on each iteration.

    .. warning::

       This algorithm may be useful for specific problems, but whether
       it will work may depend strongly on the problem.

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='excitingmixing'`` in particular.

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        Initial Jacobian approximation is (-1/alpha).
    alphamax : float, optional
        The entries of the diagonal Jacobian are kept in the range
        ``[alpha, alphamax]``.
    %(params_extra)s
    """

    def __init__(self, alpha=None, alphamax=1.0):
        GenericBroyden.__init__(self)
        self.alpha = alpha
        self.alphamax = alphamax
        self.beta = None

    def setup(self, x, F, func):
        GenericBroyden.setup(self, x, F, func)
        self.beta = np.full((self.shape[0],), self.alpha, dtype=self.dtype)

    def solve(self, f, tol=0):
        return -f * self.beta

    def matvec(self, f):
        return -f / self.beta

    def rsolve(self, f, tol=0):
        return -f * self.beta.conj()

    def rmatvec(self, f):
        return -f / self.beta.conj()

    def todense(self):
        return np.diag(-1 / self.beta)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        incr = f * self.last_f > 0
        self.beta[incr] += self.alpha
        self.beta[~incr] = self.alpha
        np.clip(self.beta, 0, self.alphamax, out=self.beta)