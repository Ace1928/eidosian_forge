import warnings
from numpy import zeros, concatenate, ravel, diff, array, ones  # noqa:F401
import numpy as np
from . import _fitpack_impl
from . import dfitpack
class RectBivariateSpline(BivariateSpline):
    """
    Bivariate spline approximation over a rectangular mesh.

    Can be used for both smoothing and interpolating data.

    Parameters
    ----------
    x,y : array_like
        1-D arrays of coordinates in strictly ascending order.
        Evaluated points outside the data range will be extrapolated.
    z : array_like
        2-D array of data with shape (x.size,y.size).
    bbox : array_like, optional
        Sequence of length 4 specifying the boundary of the rectangular
        approximation domain, which means the start and end spline knots of
        each dimension are set by these values. By default,
        ``bbox=[min(x), max(x), min(y), max(y)]``.
    kx, ky : ints, optional
        Degrees of the bivariate spline. Default is 3.
    s : float, optional
        Positive smoothing factor defined for estimation condition:
        ``sum((z[i]-f(x[i], y[i]))**2, axis=0) <= s`` where f is a spline
        function. Default is ``s=0``, which is for interpolation.

    See Also
    --------
    BivariateSpline :
        a base class for bivariate splines.
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    RectSphereBivariateSpline :
        a bivariate spline over a rectangular mesh on a sphere
    SmoothSphereBivariateSpline :
        a smoothing bivariate spline in spherical coordinates
    LSQSphereBivariateSpline :
        a bivariate spline in spherical coordinates using weighted
        least-squares fitting
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives

    Notes
    -----

    If the input data is such that input dimensions have incommensurate
    units and differ by many orders of magnitude, the interpolant may have
    numerical artifacts. Consider rescaling the data before interpolating.

    """

    def __init__(self, x, y, z, bbox=[None] * 4, kx=3, ky=3, s=0):
        x, y, bbox = (ravel(x), ravel(y), ravel(bbox))
        z = np.asarray(z)
        if not np.all(diff(x) > 0.0):
            raise ValueError('x must be strictly increasing')
        if not np.all(diff(y) > 0.0):
            raise ValueError('y must be strictly increasing')
        if not x.size == z.shape[0]:
            raise ValueError('x dimension of z must have same number of elements as x')
        if not y.size == z.shape[1]:
            raise ValueError('y dimension of z must have same number of elements as y')
        if not bbox.shape == (4,):
            raise ValueError('bbox shape should be (4,)')
        if s is not None and (not s >= 0.0):
            raise ValueError('s should be s >= 0.0')
        z = ravel(z)
        xb, xe, yb, ye = bbox
        nx, tx, ny, ty, c, fp, ier = dfitpack.regrid_smth(x, y, z, xb, xe, yb, ye, kx, ky, s)
        if ier not in [0, -1, -2]:
            msg = _surfit_messages.get(ier, 'ier=%s' % ier)
            raise ValueError(msg)
        self.fp = fp
        self.tck = (tx[:nx], ty[:ny], c[:(nx - kx - 1) * (ny - ky - 1)])
        self.degrees = (kx, ky)