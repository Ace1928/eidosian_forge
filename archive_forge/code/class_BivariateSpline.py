import warnings
from numpy import zeros, concatenate, ravel, diff, array, ones  # noqa:F401
import numpy as np
from . import _fitpack_impl
from . import dfitpack
class BivariateSpline(_BivariateSplineBase):
    """
    Base class for bivariate splines.

    This describes a spline ``s(x, y)`` of degrees ``kx`` and ``ky`` on
    the rectangle ``[xb, xe] * [yb, ye]`` calculated from a given set
    of data points ``(x, y, z)``.

    This class is meant to be subclassed, not instantiated directly.
    To construct these splines, call either `SmoothBivariateSpline` or
    `LSQBivariateSpline` or `RectBivariateSpline`.

    See Also
    --------
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
    RectBivariateSpline :
        a bivariate spline over a rectangular mesh.
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives
    """

    def ev(self, xi, yi, dx=0, dy=0):
        """
        Evaluate the spline at points

        Returns the interpolated value at ``(xi[i], yi[i]),
        i=0,...,len(xi)-1``.

        Parameters
        ----------
        xi, yi : array_like
            Input coordinates. Standard Numpy broadcasting is obeyed.
            The ordering of axes is consistent with
            ``np.meshgrid(..., indexing="ij")`` and inconsistent with the
            default ordering ``np.meshgrid(..., indexing="xy")``.
        dx : int, optional
            Order of x-derivative

            .. versionadded:: 0.14.0
        dy : int, optional
            Order of y-derivative

            .. versionadded:: 0.14.0

        Examples
        --------
        Suppose that we want to bilinearly interpolate an exponentially decaying
        function in 2 dimensions.

        >>> import numpy as np
        >>> from scipy.interpolate import RectBivariateSpline
        >>> def f(x, y):
        ...     return np.exp(-np.sqrt((x / 2) ** 2 + y**2))

        We sample the function on a coarse grid and set up the interpolator. Note that
        the default ``indexing="xy"`` of meshgrid would result in an unexpected
        (transposed) result after interpolation.

        >>> xarr = np.linspace(-3, 3, 21)
        >>> yarr = np.linspace(-3, 3, 21)
        >>> xgrid, ygrid = np.meshgrid(xarr, yarr, indexing="ij")
        >>> zdata = f(xgrid, ygrid)
        >>> rbs = RectBivariateSpline(xarr, yarr, zdata, kx=1, ky=1)

        Next we sample the function along a diagonal slice through the coordinate space
        on a finer grid using interpolation.

        >>> xinterp = np.linspace(-3, 3, 201)
        >>> yinterp = np.linspace(3, -3, 201)
        >>> zinterp = rbs.ev(xinterp, yinterp)

        And check that the interpolation passes through the function evaluations as a
        function of the distance from the origin along the slice.

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> ax1 = fig.add_subplot(1, 1, 1)
        >>> ax1.plot(np.sqrt(xarr**2 + yarr**2), np.diag(zdata), "or")
        >>> ax1.plot(np.sqrt(xinterp**2 + yinterp**2), zinterp, "-b")
        >>> plt.show()
        """
        return self.__call__(xi, yi, dx=dx, dy=dy, grid=False)

    def integral(self, xa, xb, ya, yb):
        """
        Evaluate the integral of the spline over area [xa,xb] x [ya,yb].

        Parameters
        ----------
        xa, xb : float
            The end-points of the x integration interval.
        ya, yb : float
            The end-points of the y integration interval.

        Returns
        -------
        integ : float
            The value of the resulting integral.

        """
        tx, ty, c = self.tck[:3]
        kx, ky = self.degrees
        return dfitpack.dblint(tx, ty, c, kx, ky, xa, xb, ya, yb)

    @staticmethod
    def _validate_input(x, y, z, w, kx, ky, eps):
        x, y, z = (np.asarray(x), np.asarray(y), np.asarray(z))
        if not x.size == y.size == z.size:
            raise ValueError('x, y, and z should have a same length')
        if w is not None:
            w = np.asarray(w)
            if x.size != w.size:
                raise ValueError('x, y, z, and w should have a same length')
            elif not np.all(w >= 0.0):
                raise ValueError('w should be positive')
        if eps is not None and (not 0.0 < eps < 1.0):
            raise ValueError('eps should be between (0, 1)')
        if not x.size >= (kx + 1) * (ky + 1):
            raise ValueError('The length of x, y and z should be at least (kx+1) * (ky+1)')
        return (x, y, z, w)