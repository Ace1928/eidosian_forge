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
class DiagBroyden(GenericBroyden):
    """
    Find a root of a function, using diagonal Broyden Jacobian approximation.

    The Jacobian approximation is derived from previous iterations, by
    retaining only the diagonal of Broyden matrices.

    .. warning::

       This algorithm may be useful for specific problems, but whether
       it will work may depend strongly on the problem.

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='diagbroyden'`` in particular.

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
    ...             0.5 * (x[1] - x[0])**3 + x[1]]

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.diagbroyden(fun, [0, 0])
    >>> sol
    array([0.84116403, 0.15883384])

    """

    def __init__(self, alpha=None):
        GenericBroyden.__init__(self)
        self.alpha = alpha

    def setup(self, x, F, func):
        GenericBroyden.setup(self, x, F, func)
        self.d = np.full((self.shape[0],), 1 / self.alpha, dtype=self.dtype)

    def solve(self, f, tol=0):
        return -f / self.d

    def matvec(self, f):
        return -f * self.d

    def rsolve(self, f, tol=0):
        return -f / self.d.conj()

    def rmatvec(self, f):
        return -f * self.d.conj()

    def todense(self):
        return np.diag(-self.d)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        self.d -= (df + self.d * dx) * dx / dx_norm ** 2