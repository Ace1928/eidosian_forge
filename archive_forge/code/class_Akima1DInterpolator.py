import cupy
from cupyx.scipy.interpolate._interpolate import PPoly
class Akima1DInterpolator(CubicHermiteSpline):
    """
    Akima interpolator

    Fit piecewise cubic polynomials, given vectors x and y. The interpolation
    method by Akima uses a continuously differentiable sub-spline built from
    piecewise cubic polynomials. The resultant curve passes through the given
    data points and will appear smooth and natural [1]_.

    Parameters
    ----------
    x : ndarray, shape (m, )
        1-D array of monotonically increasing real values.
    y : ndarray, shape (m, ...)
        N-D array of real values. The length of ``y`` along the first axis
        must be equal to the length of ``x``.
    axis : int, optional
        Specifies the axis of ``y`` along which to interpolate. Interpolation
        defaults to the first axis of ``y``.

    See Also
    --------
    CubicHermiteSpline : Piecewise-cubic interpolator.
    PchipInterpolator : PCHIP 1-D monotonic cubic interpolator.
    PPoly : Piecewise polynomial in terms of coefficients and breakpoints

    Notes
    -----
    Use only for precise data, as the fitted curve passes through the given
    points exactly. This routine is useful for plotting a pleasingly smooth
    curve through a few given points for purposes of plotting.

    References
    ----------
    .. [1] A new method of interpolation and smooth curve fitting based
        on local procedures. Hiroshi Akima, J. ACM, October 1970, 17(4),
        589-602.
    """

    def __init__(self, x, y, axis=0):
        x, dx, y, axis, _ = prepare_input(x, y, axis)
        m = cupy.empty((x.size + 3,) + y.shape[1:])
        dx = dx[(slice(None),) + (None,) * (y.ndim - 1)]
        m[2:-2] = cupy.diff(y, axis=0) / dx
        m[1] = 2.0 * m[2] - m[3]
        m[0] = 2.0 * m[1] - m[2]
        m[-2] = 2.0 * m[-3] - m[-4]
        m[-1] = 2.0 * m[-2] - m[-3]
        t = 0.5 * (m[3:] + m[:-3])
        dm = cupy.abs(cupy.diff(m, axis=0))
        f1 = dm[2:]
        f2 = dm[:-2]
        f12 = f1 + f2
        max_value = -cupy.inf if y.size == 0 else cupy.max(f12)
        ind = cupy.nonzero(f12 > 1e-09 * max_value)
        x_ind, y_ind = (ind[0], ind[1:])
        t[ind] = (f1[ind] * m[(x_ind + 1,) + y_ind] + f2[ind] * m[(x_ind + 2,) + y_ind]) / f12[ind]
        super().__init__(x, y, t, axis=0, extrapolate=False)
        self.axis = axis

    def extend(self, c, x, right=True):
        raise NotImplementedError('Extending a 1-D Akima interpolator is not yet implemented')

    @classmethod
    def from_spline(cls, tck, extrapolate=None):
        raise NotImplementedError('This method does not make sense for an Akima interpolator.')

    @classmethod
    def from_bernstein_basis(cls, bp, extrapolate=None):
        raise NotImplementedError('This method does not make sense for an Akima interpolator.')