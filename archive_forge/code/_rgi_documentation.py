import itertools
import cupy as cp
from cupyx.scipy.interpolate._bspline2 import make_interp_spline
from cupyx.scipy.interpolate._cubic import PchipInterpolator

        Interpolation at coordinates.

        Parameters
        ----------
        xi : cupy.ndarray of shape (..., ndim)
            The coordinates to evaluate the interpolator at.

        method : str, optional
            The method of interpolation to perform. Supported are "linear" and
            "nearest".  Default is the method chosen when the interpolator was
            created.

        Returns
        -------
        values_x : cupy.ndarray, shape xi.shape[:-1] + values.shape[ndim:]
            Interpolated values at `xi`. See notes for behaviour when
            ``xi.ndim == 1``.

        Notes
        -----
        In the case that ``xi.ndim == 1`` a new axis is inserted into
        the 0 position of the returned array, values_x, so its shape is
        instead ``(1,) + values.shape[ndim:]``.

        Examples
        --------
        Here we define a nearest-neighbor interpolator of a simple function

        >>> import cupy as cp
        >>> x, y = cp.array([0, 1, 2]), cp.array([1, 3, 7])
        >>> def f(x, y):
        ...     return x**2 + y**2
        >>> data = f(*cp.meshgrid(x, y, indexing='ij', sparse=True))
        >>> from cupyx.scipy.interpolate import RegularGridInterpolator
        >>> interp = RegularGridInterpolator((x, y), data, method='nearest')

        By construction, the interpolator uses the nearest-neighbor
        interpolation

        >>> interp([[1.5, 1.3], [0.3, 4.5]])
        array([2., 9.])

        We can however evaluate the linear interpolant by overriding the
        `method` parameter

        >>> interp([[1.5, 1.3], [0.3, 4.5]], method='linear')
        array([ 4.7, 24.3])
        