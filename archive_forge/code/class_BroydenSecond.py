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
class BroydenSecond(BroydenFirst):
    """
    Find a root of a function, using Broyden's second Jacobian approximation.

    This method is also known as "Broyden's bad method".

    Parameters
    ----------
    %(params_basic)s
    %(broyden_params)s
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='broyden2'`` in particular.

    Notes
    -----
    This algorithm implements the inverse Jacobian Quasi-Newton update

    .. math:: H_+ = H + (dx - H df) df^\\dagger / ( df^\\dagger df)

    corresponding to Broyden's second method.

    References
    ----------
    .. [1] B.A. van der Rotten, PhD thesis,
       "A limited memory Broyden method to solve high-dimensional
       systems of nonlinear equations". Mathematisch Instituut,
       Universiteit Leiden, The Netherlands (2003).

       https://web.archive.org/web/20161022015821/http://www.math.leidenuniv.nl/scripties/Rotten.pdf

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
    ...             0.5 * (x[1] - x[0])**3 + x[1]]

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.broyden2(fun, [0, 0])
    >>> sol
    array([0.84116365, 0.15883529])

    """

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        self._reduce()
        v = df
        c = dx - self.Gm.matvec(df)
        d = v / df_norm ** 2
        self.Gm.append(c, d)