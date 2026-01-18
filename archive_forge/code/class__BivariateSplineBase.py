import warnings
from numpy import zeros, concatenate, ravel, diff, array, ones  # noqa:F401
import numpy as np
from . import _fitpack_impl
from . import dfitpack
class _BivariateSplineBase:
    """ Base class for Bivariate spline s(x,y) interpolation on the rectangle
    [xb,xe] x [yb, ye] calculated from a given set of data points
    (x,y,z).

    See Also
    --------
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives
    BivariateSpline :
        a base class for bivariate splines.
    SphereBivariateSpline :
        a bivariate spline on a spherical grid
    """

    @classmethod
    def _from_tck(cls, tck):
        """Construct a spline object from given tck and degree"""
        self = cls.__new__(cls)
        if len(tck) != 5:
            raise ValueError('tck should be a 5 element tuple of tx, ty, c, kx, ky')
        self.tck = tck[:3]
        self.degrees = tck[3:]
        return self

    def get_residual(self):
        """ Return weighted sum of squared residuals of the spline
        approximation: sum ((w[i]*(z[i]-s(x[i],y[i])))**2,axis=0)
        """
        return self.fp

    def get_knots(self):
        """ Return a tuple (tx,ty) where tx,ty contain knots positions
        of the spline with respect to x-, y-variable, respectively.
        The position of interior and additional knots are given as
        t[k+1:-k-1] and t[:k+1]=b, t[-k-1:]=e, respectively.
        """
        return self.tck[:2]

    def get_coeffs(self):
        """ Return spline coefficients."""
        return self.tck[2]

    def __call__(self, x, y, dx=0, dy=0, grid=True):
        """
        Evaluate the spline or its derivatives at given positions.

        Parameters
        ----------
        x, y : array_like
            Input coordinates.

            If `grid` is False, evaluate the spline at points ``(x[i],
            y[i]), i=0, ..., len(x)-1``.  Standard Numpy broadcasting
            is obeyed.

            If `grid` is True: evaluate spline at the grid points
            defined by the coordinate arrays x, y. The arrays must be
            sorted to increasing order.

            The ordering of axes is consistent with
            ``np.meshgrid(..., indexing="ij")`` and inconsistent with the
            default ordering ``np.meshgrid(..., indexing="xy")``.
        dx : int
            Order of x-derivative

            .. versionadded:: 0.14.0
        dy : int
            Order of y-derivative

            .. versionadded:: 0.14.0
        grid : bool
            Whether to evaluate the results on a grid spanned by the
            input arrays, or at points specified by the input arrays.

            .. versionadded:: 0.14.0

        Examples
        --------
        Suppose that we want to bilinearly interpolate an exponentially decaying
        function in 2 dimensions.

        >>> import numpy as np
        >>> from scipy.interpolate import RectBivariateSpline

        We sample the function on a coarse grid. Note that the default indexing="xy"
        of meshgrid would result in an unexpected (transposed) result after
        interpolation.

        >>> xarr = np.linspace(-3, 3, 100)
        >>> yarr = np.linspace(-3, 3, 100)
        >>> xgrid, ygrid = np.meshgrid(xarr, yarr, indexing="ij")

        The function to interpolate decays faster along one axis than the other.

        >>> zdata = np.exp(-np.sqrt((xgrid / 2) ** 2 + ygrid**2))

        Next we sample on a finer grid using interpolation (kx=ky=1 for bilinear).

        >>> rbs = RectBivariateSpline(xarr, yarr, zdata, kx=1, ky=1)
        >>> xarr_fine = np.linspace(-3, 3, 200)
        >>> yarr_fine = np.linspace(-3, 3, 200)
        >>> xgrid_fine, ygrid_fine = np.meshgrid(xarr_fine, yarr_fine, indexing="ij")
        >>> zdata_interp = rbs(xgrid_fine, ygrid_fine, grid=False)

        And check that the result agrees with the input by plotting both.

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> ax1 = fig.add_subplot(1, 2, 1, aspect="equal")
        >>> ax2 = fig.add_subplot(1, 2, 2, aspect="equal")
        >>> ax1.imshow(zdata)
        >>> ax2.imshow(zdata_interp)
        >>> plt.show()
        """
        x = np.asarray(x)
        y = np.asarray(y)
        tx, ty, c = self.tck[:3]
        kx, ky = self.degrees
        if grid:
            if x.size == 0 or y.size == 0:
                return np.zeros((x.size, y.size), dtype=self.tck[2].dtype)
            if x.size >= 2 and (not np.all(np.diff(x) >= 0.0)):
                raise ValueError('x must be strictly increasing when `grid` is True')
            if y.size >= 2 and (not np.all(np.diff(y) >= 0.0)):
                raise ValueError('y must be strictly increasing when `grid` is True')
            if dx or dy:
                z, ier = dfitpack.parder(tx, ty, c, kx, ky, dx, dy, x, y)
                if not ier == 0:
                    raise ValueError('Error code returned by parder: %s' % ier)
            else:
                z, ier = dfitpack.bispev(tx, ty, c, kx, ky, x, y)
                if not ier == 0:
                    raise ValueError('Error code returned by bispev: %s' % ier)
        else:
            if x.shape != y.shape:
                x, y = np.broadcast_arrays(x, y)
            shape = x.shape
            x = x.ravel()
            y = y.ravel()
            if x.size == 0 or y.size == 0:
                return np.zeros(shape, dtype=self.tck[2].dtype)
            if dx or dy:
                z, ier = dfitpack.pardeu(tx, ty, c, kx, ky, dx, dy, x, y)
                if not ier == 0:
                    raise ValueError('Error code returned by pardeu: %s' % ier)
            else:
                z, ier = dfitpack.bispeu(tx, ty, c, kx, ky, x, y)
                if not ier == 0:
                    raise ValueError('Error code returned by bispeu: %s' % ier)
            z = z.reshape(shape)
        return z

    def partial_derivative(self, dx, dy):
        """Construct a new spline representing a partial derivative of this
        spline.

        Parameters
        ----------
        dx, dy : int
            Orders of the derivative in x and y respectively. They must be
            non-negative integers and less than the respective degree of the
            original spline (self) in that direction (``kx``, ``ky``).

        Returns
        -------
        spline :
            A new spline of degrees (``kx - dx``, ``ky - dy``) representing the
            derivative of this spline.

        Notes
        -----

        .. versionadded:: 1.9.0

        """
        if dx == 0 and dy == 0:
            return self
        else:
            kx, ky = self.degrees
            if not (dx >= 0 and dy >= 0):
                raise ValueError('order of derivative must be positive or zero')
            if not (dx < kx and dy < ky):
                raise ValueError('order of derivative must be less than degree of spline')
            tx, ty, c = self.tck[:3]
            newc, ier = dfitpack.pardtc(tx, ty, c, kx, ky, dx, dy)
            if ier != 0:
                raise ValueError('Unexpected error code returned by pardtc: %d' % ier)
            nx = len(tx)
            ny = len(ty)
            newtx = tx[dx:nx - dx]
            newty = ty[dy:ny - dy]
            newkx, newky = (kx - dx, ky - dy)
            newclen = (nx - dx - kx - 1) * (ny - dy - ky - 1)
            return _DerivedBivariateSpline._from_tck((newtx, newty, newc[:newclen], newkx, newky))