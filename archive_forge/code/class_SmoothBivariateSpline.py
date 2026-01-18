import warnings
from numpy import zeros, concatenate, ravel, diff, array, ones  # noqa:F401
import numpy as np
from . import _fitpack_impl
from . import dfitpack
class SmoothBivariateSpline(BivariateSpline):
    """
    Smooth bivariate spline approximation.

    Parameters
    ----------
    x, y, z : array_like
        1-D sequences of data points (order is not important).
    w : array_like, optional
        Positive 1-D sequence of weights, of same length as `x`, `y` and `z`.
    bbox : array_like, optional
        Sequence of length 4 specifying the boundary of the rectangular
        approximation domain.  By default,
        ``bbox=[min(x), max(x), min(y), max(y)]``.
    kx, ky : ints, optional
        Degrees of the bivariate spline. Default is 3.
    s : float, optional
        Positive smoothing factor defined for estimation condition:
        ``sum((w[i]*(z[i]-s(x[i], y[i])))**2, axis=0) <= s``
        Default ``s=len(w)`` which should be a good value if ``1/w[i]`` is an
        estimate of the standard deviation of ``z[i]``.
    eps : float, optional
        A threshold for determining the effective rank of an over-determined
        linear system of equations. `eps` should have a value within the open
        interval ``(0, 1)``, the default is 1e-16.

    See Also
    --------
    BivariateSpline :
        a base class for bivariate splines.
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    RectSphereBivariateSpline :
        a bivariate spline over a rectangular mesh on a sphere
    SmoothSphereBivariateSpline :
        a smoothing bivariate spline in spherical coordinates
    LSQSphereBivariateSpline :
        a bivariate spline in spherical coordinates using weighted
        least-squares fitting
    RectBivariateSpline :
        a bivariate spline over a rectangular mesh
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives

    Notes
    -----
    The length of `x`, `y` and `z` should be at least ``(kx+1) * (ky+1)``.

    If the input data is such that input dimensions have incommensurate
    units and differ by many orders of magnitude, the interpolant may have
    numerical artifacts. Consider rescaling the data before interpolating.

    This routine constructs spline knot vectors automatically via the FITPACK
    algorithm. The spline knots may be placed away from the data points. For
    some data sets, this routine may fail to construct an interpolating spline,
    even if one is requested via ``s=0`` parameter. In such situations, it is
    recommended to use `bisplrep` / `bisplev` directly instead of this routine
    and, if needed, increase the values of ``nxest`` and ``nyest`` parameters
    of `bisplrep`.

    For linear interpolation, prefer `LinearNDInterpolator`.
    See ``https://gist.github.com/ev-br/8544371b40f414b7eaf3fe6217209bff``
    for discussion.

    """

    def __init__(self, x, y, z, w=None, bbox=[None] * 4, kx=3, ky=3, s=None, eps=1e-16):
        x, y, z, w = self._validate_input(x, y, z, w, kx, ky, eps)
        bbox = ravel(bbox)
        if not bbox.shape == (4,):
            raise ValueError('bbox shape should be (4,)')
        if s is not None and (not s >= 0.0):
            raise ValueError('s should be s >= 0.0')
        xb, xe, yb, ye = bbox
        nx, tx, ny, ty, c, fp, wrk1, ier = dfitpack.surfit_smth(x, y, z, w, xb, xe, yb, ye, kx, ky, s=s, eps=eps, lwrk2=1)
        if ier > 10:
            nx, tx, ny, ty, c, fp, wrk1, ier = dfitpack.surfit_smth(x, y, z, w, xb, xe, yb, ye, kx, ky, s=s, eps=eps, lwrk2=ier)
        if ier in [0, -1, -2]:
            pass
        else:
            message = _surfit_messages.get(ier, 'ier=%s' % ier)
            warnings.warn(message, stacklevel=2)
        self.fp = fp
        self.tck = (tx[:nx], ty[:ny], c[:(nx - kx - 1) * (ny - ky - 1)])
        self.degrees = (kx, ky)