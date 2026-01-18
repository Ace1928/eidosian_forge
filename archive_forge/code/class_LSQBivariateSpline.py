import warnings
from numpy import zeros, concatenate, ravel, diff, array, ones  # noqa:F401
import numpy as np
from . import _fitpack_impl
from . import dfitpack
class LSQBivariateSpline(BivariateSpline):
    """
    Weighted least-squares bivariate spline approximation.

    Parameters
    ----------
    x, y, z : array_like
        1-D sequences of data points (order is not important).
    tx, ty : array_like
        Strictly ordered 1-D sequences of knots coordinates.
    w : array_like, optional
        Positive 1-D array of weights, of the same length as `x`, `y` and `z`.
    bbox : (4,) array_like, optional
        Sequence of length 4 specifying the boundary of the rectangular
        approximation domain.  By default,
        ``bbox=[min(x,tx),max(x,tx), min(y,ty),max(y,ty)]``.
    kx, ky : ints, optional
        Degrees of the bivariate spline. Default is 3.
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
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    RectSphereBivariateSpline :
        a bivariate spline over a rectangular mesh on a sphere
    SmoothSphereBivariateSpline :
        a smoothing bivariate spline in spherical coordinates
    LSQSphereBivariateSpline :
        a bivariate spline in spherical coordinates using weighted
        least-squares fitting
    RectBivariateSpline :
        a bivariate spline over a rectangular mesh.
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

    """

    def __init__(self, x, y, z, tx, ty, w=None, bbox=[None] * 4, kx=3, ky=3, eps=None):
        x, y, z, w = self._validate_input(x, y, z, w, kx, ky, eps)
        bbox = ravel(bbox)
        if not bbox.shape == (4,):
            raise ValueError('bbox shape should be (4,)')
        nx = 2 * kx + 2 + len(tx)
        ny = 2 * ky + 2 + len(ty)
        nmax = max(nx, ny)
        tx1 = zeros((nmax,), float)
        ty1 = zeros((nmax,), float)
        tx1[kx + 1:nx - kx - 1] = tx
        ty1[ky + 1:ny - ky - 1] = ty
        xb, xe, yb, ye = bbox
        tx1, ty1, c, fp, ier = dfitpack.surfit_lsq(x, y, z, nx, tx1, ny, ty1, w, xb, xe, yb, ye, kx, ky, eps, lwrk2=1)
        if ier > 10:
            tx1, ty1, c, fp, ier = dfitpack.surfit_lsq(x, y, z, nx, tx1, ny, ty1, w, xb, xe, yb, ye, kx, ky, eps, lwrk2=ier)
        if ier in [0, -1, -2]:
            pass
        else:
            if ier < -2:
                deficiency = (nx - kx - 1) * (ny - ky - 1) + ier
                message = _surfit_messages.get(-3) % deficiency
            else:
                message = _surfit_messages.get(ier, 'ier=%s' % ier)
            warnings.warn(message, stacklevel=2)
        self.fp = fp
        self.tck = (tx1[:nx], ty1[:ny], c)
        self.degrees = (kx, ky)