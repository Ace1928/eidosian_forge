import math
import cupy
from cupy._core import internal  # NOQA
from cupy._core._scalar import get_typename  # NOQA
from cupy_backends.cuda.api import runtime
from cupyx.scipy import special as spec
from cupyx.scipy.interpolate._bspline import BSpline, _get_dtype
import numpy as np
class PPoly(_PPolyBase):
    """
    Piecewise polynomial in terms of coefficients and breakpoints
    The polynomial between ``x[i]`` and ``x[i + 1]`` is written in the
    local power basis::

        S = sum(c[m, i] * (xp - x[i]) ** (k - m) for m in range(k + 1))

    where ``k`` is the degree of the polynomial.

    Parameters
    ----------
    c : ndarray, shape (k, m, ...)
        Polynomial coefficients, order `k` and `m` intervals.
    x : ndarray, shape (m+1,)
        Polynomial breakpoints. Must be sorted in either increasing or
        decreasing order.
    extrapolate : bool or 'periodic', optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. Default is True.
    axis : int, optional
        Interpolation axis. Default is zero.

    Attributes
    ----------
    x : ndarray
        Breakpoints.
    c : ndarray
        Coefficients of the polynomials. They are reshaped
        to a 3-D array with the last dimension representing
        the trailing dimensions of the original coefficient array.
    axis : int
        Interpolation axis.

    See also
    --------
    BPoly : piecewise polynomials in the Bernstein basis

    Notes
    -----
    High-order polynomials in the power basis can be numerically
    unstable. Precision problems can start to appear for orders
    larger than 20-30.

    .. seealso:: :class:`scipy.interpolate.BSpline`
    """

    def _evaluate(self, x, nu, extrapolate, out):
        _ppoly_evaluate(self.c.reshape(self.c.shape[0], self.c.shape[1], -1), self.x, x, nu, bool(extrapolate), out)

    def derivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the derivative.

        Parameters
        ----------
        nu : int, optional
            Order of derivative to evaluate. Default is 1, i.e., compute the
            first derivative. If negative, the antiderivative is returned.

        Returns
        -------
        pp : PPoly
            Piecewise polynomial of order k2 = k - n representing the
            derivative of this polynomial.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.
        """
        if nu < 0:
            return self.antiderivative(-nu)
        if nu == 0:
            c2 = self.c.copy()
        else:
            c2 = self.c[:-nu, :].copy()
        if c2.shape[0] == 0:
            c2 = cupy.zeros((1,) + c2.shape[1:], dtype=c2.dtype)
        factor = spec.poch(cupy.arange(c2.shape[0], 0, -1), nu)
        c2 *= factor[(slice(None),) + (None,) * (c2.ndim - 1)]
        return self.construct_fast(c2, self.x, self.extrapolate, self.axis)

    def antiderivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the antiderivative.
        Antiderivative is also the indefinite integral of the function,
        and derivative is its inverse operation.

        Parameters
        ----------
        nu : int, optional
            Order of antiderivative to evaluate. Default is 1, i.e., compute
            the first integral. If negative, the derivative is returned.

        Returns
        -------
        pp : PPoly
            Piecewise polynomial of order k2 = k + n representing
            the antiderivative of this polynomial.

        Notes
        -----
        The antiderivative returned by this function is continuous and
        continuously differentiable to order n-1, up to floating point
        rounding error.

        If antiderivative is computed and ``self.extrapolate='periodic'``,
        it will be set to False for the returned instance. This is done because
        the antiderivative is no longer periodic and its correct evaluation
        outside of the initially given x interval is difficult.
        """
        if nu <= 0:
            return self.derivative(-nu)
        c = cupy.zeros((self.c.shape[0] + nu, self.c.shape[1]) + self.c.shape[2:], dtype=self.c.dtype)
        c[:-nu] = self.c
        factor = spec.poch(cupy.arange(self.c.shape[0], 0, -1), nu)
        c[:-nu] /= factor[(slice(None),) + (None,) * (c.ndim - 1)]
        self._ensure_c_contiguous()
        _fix_continuity(c.reshape(c.shape[0], c.shape[1], -1), self.x, nu - 1)
        if self.extrapolate == 'periodic':
            extrapolate = False
        else:
            extrapolate = self.extrapolate
        return self.construct_fast(c, self.x, extrapolate, self.axis)

    def integrate(self, a, b, extrapolate=None):
        """
        Compute a definite integral over a piecewise polynomial.

        Parameters
        ----------
        a : float
            Lower integration bound
        b : float
            Upper integration bound
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used.
            If None (default), use `self.extrapolate`.

        Returns
        -------
        ig : array_like
            Definite integral of the piecewise polynomial over [a, b]
        """
        if extrapolate is None:
            extrapolate = self.extrapolate
        sign = 1
        if b < a:
            a, b = (b, a)
            sign = -1
        range_int = cupy.empty((int(np.prod(self.c.shape[2:])),), dtype=self.c.dtype)
        self._ensure_c_contiguous()
        if extrapolate == 'periodic':
            xs, xe = (self.x[0], self.x[-1])
            period = xe - xs
            interval = b - a
            n_periods, left = divmod(interval, period)
            if n_periods > 0:
                _integrate(self.c.reshape(self.c.shape[0], self.c.shape[1], -1), self.x, xs, xe, False, out=range_int)
                range_int *= n_periods
            else:
                range_int.fill(0)
            a = xs + (a - xs) % period
            b = a + left
            remainder_int = cupy.empty_like(range_int)
            if b <= xe:
                _integrate(self.c.reshape(self.c.shape[0], self.c.shape[1], -1), self.x, a, b, False, out=remainder_int)
                range_int += remainder_int
            else:
                _integrate(self.c.reshape(self.c.shape[0], self.c.shape[1], -1), self.x, a, xe, False, out=remainder_int)
                range_int += remainder_int
                _integrate(self.c.reshape(self.c.shape[0], self.c.shape[1], -1), self.x, xs, xs + left + a - xe, False, out=remainder_int)
                range_int += remainder_int
        else:
            _integrate(self.c.reshape(self.c.shape[0], self.c.shape[1], -1), self.x, a, b, bool(extrapolate), out=range_int)
        range_int *= sign
        return range_int.reshape(self.c.shape[2:])

    def solve(self, y=0.0, discontinuity=True, extrapolate=None):
        """
        Find real solutions of the equation ``pp(x) == y``.

        Parameters
        ----------
        y : float, optional
            Right-hand side. Default is zero.
        discontinuity : bool, optional
            Whether to report sign changes across discontinuities at
            breakpoints as roots.
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to return roots from the polynomial
            extrapolated based on first and last intervals, 'periodic' works
            the same as False. If None (default), use `self.extrapolate`.

        Returns
        -------
        roots : ndarray
            Roots of the polynomial(s).
            If the PPoly object describes multiple polynomials, the
            return value is an object array whose each element is an
            ndarray containing the roots.

        Notes
        -----
        This routine works only on real-valued polynomials.
        If the piecewise polynomial contains sections that are
        identically zero, the root list will contain the start point
        of the corresponding interval, followed by a ``nan`` value.
        If the polynomial is discontinuous across a breakpoint, and
        there is a sign change across the breakpoint, this is reported
        if the `discont` parameter is True.

        At the moment, there is not an actual implementation.
        """
        raise NotImplementedError('At the moment there is not a GPU implementation for solve')

    def roots(self, discontinuity=True, extrapolate=None):
        """
        Find real roots of the piecewise polynomial.

        Parameters
        ----------
        discontinuity : bool, optional
            Whether to report sign changes across discontinuities at
            breakpoints as roots.
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to return roots from the polynomial
            extrapolated based on first and last intervals, 'periodic' works
            the same as False. If None (default), use `self.extrapolate`.

        Returns
        -------
        roots : ndarray
            Roots of the polynomial(s).
            If the PPoly object describes multiple polynomials, the
            return value is an object array whose each element is an
            ndarray containing the roots.

        See Also
        --------
        PPoly.solve
        """
        return self.solve(0, discontinuity, extrapolate)

    @classmethod
    def from_spline(cls, tck, extrapolate=None):
        """
        Construct a piecewise polynomial from a spline

        Parameters
        ----------
        tck
            A spline, as a (knots, coefficients, degree) tuple or
            a BSpline object.
        extrapolate : bool or 'periodic', optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used. Default is True.
        """
        if isinstance(tck, BSpline):
            t, c, k = tck.tck
            if extrapolate is None:
                extrapolate = tck.extrapolate
        else:
            t, c, k = tck
        spl = BSpline(t, c, k, extrapolate=extrapolate)
        cvals = cupy.empty((k + 1, len(t) - 1), dtype=c.dtype)
        for m in range(k, -1, -1):
            y = spl(t[:-1], nu=m)
            cvals[k - m, :] = y / spec.gamma(m + 1)
        return cls.construct_fast(cvals, t, extrapolate)

    @classmethod
    def from_bernstein_basis(cls, bp, extrapolate=None):
        """
        Construct a piecewise polynomial in the power basis
        from a polynomial in Bernstein basis.

        Parameters
        ----------
        bp : BPoly
            A Bernstein basis polynomial, as created by BPoly
        extrapolate : bool or 'periodic', optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used. Default is True.
        """
        if not isinstance(bp, BPoly):
            raise TypeError('.from_bernstein_basis only accepts BPoly instances. Got %s instead.' % type(bp))
        dx = cupy.diff(bp.x)
        k = bp.c.shape[0] - 1
        rest = (None,) * (bp.c.ndim - 2)
        c = cupy.zeros_like(bp.c)
        for a in range(k + 1):
            factor = (-1) ** a * _comb(k, a) * bp.c[a]
            for s in range(a, k + 1):
                val = _comb(k - a, s - a) * (-1) ** s
                c[k - s] += factor * val / dx[(slice(None),) + rest] ** s
        if extrapolate is None:
            extrapolate = bp.extrapolate
        return cls.construct_fast(c, bp.x, extrapolate, bp.axis)