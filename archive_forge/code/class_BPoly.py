import math
import cupy
from cupy._core import internal  # NOQA
from cupy._core._scalar import get_typename  # NOQA
from cupy_backends.cuda.api import runtime
from cupyx.scipy import special as spec
from cupyx.scipy.interpolate._bspline import BSpline, _get_dtype
import numpy as np
class BPoly(_PPolyBase):
    """
    Piecewise polynomial in terms of coefficients and breakpoints.

    The polynomial between ``x[i]`` and ``x[i + 1]`` is written in the

    Bernstein polynomial basis::

        S = sum(c[a, i] * b(a, k; x) for a in range(k+1)),

    where ``k`` is the degree of the polynomial, and::

        b(a, k; x) = binom(k, a) * t**a * (1 - t)**(k - a),

    with ``t = (x - x[i]) / (x[i+1] - x[i])`` and ``binom`` is the binomial
    coefficient.

    Parameters
    ----------
    c : ndarray, shape (k, m, ...)
        Polynomial coefficients, order `k` and `m` intervals
    x : ndarray, shape (m+1,)
        Polynomial breakpoints. Must be sorted in either increasing or
        decreasing order.
    extrapolate : bool, optional
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
    PPoly : piecewise polynomials in the power basis

    Notes
    -----
    Properties of Bernstein polynomials are well documented in the literature,
    see for example [1]_ [2]_ [3]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Bernstein_polynomial
    .. [2] Kenneth I. Joy, Bernstein polynomials,
       http://www.idav.ucdavis.edu/education/CAGDNotes/Bernstein-Polynomials.pdf
    .. [3] E. H. Doha, A. H. Bhrawy, and M. A. Saker, Boundary Value Problems,
           vol 2011, article ID 829546,
           `10.1155/2011/829543 <https://doi.org/10.1155/2011/829543>`_.

    Examples
    --------
    >>> from cupyx.scipy.interpolate import BPoly
    >>> x = [0, 1]
    >>> c = [[1], [2], [3]]
    >>> bp = BPoly(c, x)

    This creates a 2nd order polynomial

    .. math::

        B(x) = 1 \\times b_{0, 2}(x) + 2 \\times b_{1, 2}(x) +
               3 \\times b_{2, 2}(x) \\\\
             = 1 \\times (1-x)^2 + 2 \\times 2 x (1 - x) + 3 \\times x^2
    """

    def _evaluate(self, x, nu, extrapolate, out):
        if nu < 0:
            raise NotImplementedError('Cannot do antiderivatives in the B-basis yet.')
        _bpoly_evaluate(self.c.reshape(self.c.shape[0], self.c.shape[1], -1), self.x, x, nu, bool(extrapolate), out)

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
        bp : BPoly
            Piecewise polynomial of order k - nu representing the derivative of
            this polynomial.
        """
        if nu < 0:
            return self.antiderivative(-nu)
        if nu > 1:
            bp = self
            for k in range(nu):
                bp = bp.derivative()
            return bp
        if nu == 0:
            c2 = self.c.copy()
        else:
            rest = (None,) * (self.c.ndim - 2)
            k = self.c.shape[0] - 1
            dx = cupy.diff(self.x)[(None, slice(None)) + rest]
            c2 = k * cupy.diff(self.c, axis=0) / dx
        if c2.shape[0] == 0:
            c2 = cupy.zeros((1,) + c2.shape[1:], dtype=c2.dtype)
        return self.construct_fast(c2, self.x, self.extrapolate, self.axis)

    def antiderivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the antiderivative.

        Parameters
        ----------
        nu : int, optional
            Order of antiderivative to evaluate. Default is 1, i.e., compute
            the first integral. If negative, the derivative is returned.

        Returns
        -------
        bp : BPoly
            Piecewise polynomial of order k + nu representing the
            antiderivative of this polynomial.

        Notes
        -----
        If antiderivative is computed and ``self.extrapolate='periodic'``,
        it will be set to False for the returned instance. This is done because
        the antiderivative is no longer periodic and its correct evaluation
        outside of the initially given x interval is difficult.
        """
        if nu <= 0:
            return self.derivative(-nu)
        if nu > 1:
            bp = self
            for k in range(nu):
                bp = bp.antiderivative()
            return bp
        c, x = (self.c, self.x)
        k = c.shape[0]
        c2 = cupy.zeros((k + 1,) + c.shape[1:], dtype=c.dtype)
        c2[1:, ...] = cupy.cumsum(c, axis=0) / k
        delta = x[1:] - x[:-1]
        c2 *= delta[(None, slice(None)) + (None,) * (c.ndim - 2)]
        c2[:, 1:] += cupy.cumsum(c2[k, :], axis=0)[:-1]
        if self.extrapolate == 'periodic':
            extrapolate = False
        else:
            extrapolate = self.extrapolate
        return self.construct_fast(c2, x, extrapolate, axis=self.axis)

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
            Whether to extrapolate to out-of-bounds points based on first
            and last intervals, or to return NaNs. If 'periodic', periodic
            extrapolation is used. If None (default), use `self.extrapolate`.

        Returns
        -------
        array_like
            Definite integral of the piecewise polynomial over [a, b]
        """
        ib = self.antiderivative()
        if extrapolate is None:
            extrapolate = self.extrapolate
        if extrapolate != 'periodic':
            ib.extrapolate = extrapolate
        if extrapolate == 'periodic':
            if a <= b:
                sign = 1
            else:
                a, b = (b, a)
                sign = -1
            xs, xe = (self.x[0], self.x[-1])
            period = xe - xs
            interval = b - a
            n_periods, left = divmod(interval, period)
            res = n_periods * (ib(xe) - ib(xs))
            a = xs + (a - xs) % period
            b = a + left
            if b <= xe:
                res += ib(b) - ib(a)
            else:
                res += ib(xe) - ib(a) + ib(xs + left + a - xe) - ib(xs)
            return sign * res
        else:
            return ib(b) - ib(a)

    def extend(self, c, x):
        k = max(self.c.shape[0], c.shape[0])
        self.c = self._raise_degree(self.c, k - self.c.shape[0])
        c = self._raise_degree(c, k - c.shape[0])
        return _PPolyBase.extend(self, c, x)
    extend.__doc__ = _PPolyBase.extend.__doc__

    @staticmethod
    def _raise_degree(c, d):
        """
        Raise a degree of a polynomial in the Bernstein basis.

        Given the coefficients of a polynomial degree `k`, return (the
        coefficients of) the equivalent polynomial of degree `k+d`.

        Parameters
        ----------
        c : array_like
            coefficient array, 1-D
        d : integer

        Returns
        -------
        array
            coefficient array, 1-D array of length `c.shape[0] + d`

        Notes
        -----
        This uses the fact that a Bernstein polynomial `b_{a, k}` can be
        identically represented as a linear combination of polynomials of
        a higher degree `k+d`:

            .. math:: b_{a, k} = comb(k, a) \\sum_{j=0}^{d} b_{a+j, k+d} \\
                                 comb(d, j) / comb(k+d, a+j)
        """
        if d == 0:
            return c
        k = c.shape[0] - 1
        out = cupy.zeros((c.shape[0] + d,) + c.shape[1:], dtype=c.dtype)
        for a in range(c.shape[0]):
            f = c[a] * _comb(k, a)
            for j in range(d + 1):
                out[a + j] += f * _comb(d, j) / _comb(k + d, a + j)
        return out

    @classmethod
    def from_power_basis(cls, pp, extrapolate=None):
        """
        Construct a piecewise polynomial in Bernstein basis
        from a power basis polynomial.

        Parameters
        ----------
        pp : PPoly
            A piecewise polynomial in the power basis
        extrapolate : bool or 'periodic', optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used. Default is True.
        """
        if not isinstance(pp, PPoly):
            raise TypeError('.from_power_basis only accepts PPoly instances. Got %s instead.' % type(pp))
        dx = cupy.diff(pp.x)
        k = pp.c.shape[0] - 1
        rest = (None,) * (pp.c.ndim - 2)
        c = cupy.zeros_like(pp.c)
        for a in range(k + 1):
            factor = pp.c[a] / _comb(k, k - a) * dx[(slice(None),) + rest] ** (k - a)
            for j in range(k - a, k + 1):
                c[j] += factor * _comb(j, k - a)
        if extrapolate is None:
            extrapolate = pp.extrapolate
        return cls.construct_fast(c, pp.x, extrapolate, pp.axis)

    @classmethod
    def from_derivatives(cls, xi, yi, orders=None, extrapolate=None):
        """
        Construct a piecewise polynomial in the Bernstein basis,
        compatible with the specified values and derivatives at breakpoints.

        Parameters
        ----------
        xi : array_like
            sorted 1-D array of x-coordinates
        yi : array_like or list of array_likes
            ``yi[i][j]`` is the ``j`` th derivative known at ``xi[i]``
        orders : None or int or array_like of ints. Default: None.
            Specifies the degree of local polynomials. If not None, some
            derivatives are ignored.
        extrapolate : bool or 'periodic', optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used. Default is True.

        Notes
        -----
        If ``k`` derivatives are specified at a breakpoint ``x``, the
        constructed polynomial is exactly ``k`` times continuously
        differentiable at ``x``, unless the ``order`` is provided explicitly.
        In the latter case, the smoothness of the polynomial at
        the breakpoint is controlled by the ``order``.

        Deduces the number of derivatives to match at each end
        from ``order`` and the number of derivatives available. If
        possible it uses the same number of derivatives from
        each end; if the number is odd it tries to take the
        extra one from y2. In any case if not enough derivatives
        are available at one end or another it draws enough to
        make up the total from the other end.

        If the order is too high and not enough derivatives are available,
        an exception is raised.

        Examples
        --------
        >>> from cupyx.scipy.interpolate import BPoly
        >>> BPoly.from_derivatives([0, 1], [[1, 2], [3, 4]])

        Creates a polynomial `f(x)` of degree 3, defined on `[0, 1]`
        such that `f(0) = 1, df/dx(0) = 2, f(1) = 3, df/dx(1) = 4`

        >>> BPoly.from_derivatives([0, 1, 2], [[0, 1], [0], [2]])

        Creates a piecewise polynomial `f(x)`, such that
        `f(0) = f(1) = 0`, `f(2) = 2`, and `df/dx(0) = 1`.
        Based on the number of derivatives provided, the order of the
        local polynomials is 2 on `[0, 1]` and 1 on `[1, 2]`.
        Notice that no restriction is imposed on the derivatives at
        ``x = 1`` and ``x = 2``.

        Indeed, the explicit form of the polynomial is::

            f(x) = | x * (1 - x),  0 <= x < 1
                   | 2 * (x - 1),  1 <= x <= 2

        So that f'(1-0) = -1 and f'(1+0) = 2
        """
        xi = cupy.asarray(xi)
        if len(xi) != len(yi):
            raise ValueError('xi and yi need to have the same length')
        if cupy.any(xi[1:] - xi[:1] <= 0):
            raise ValueError('x coordinates are not in increasing order')
        m = len(xi) - 1
        try:
            k = max((len(yi[i]) + len(yi[i + 1]) for i in range(m)))
        except TypeError as e:
            raise ValueError('Using a 1-D array for y? Please .reshape(-1, 1).') from e
        if orders is None:
            orders = [None] * m
        else:
            if isinstance(orders, (int, cupy.integer)):
                orders = [orders] * m
            k = max(k, max(orders))
            if any((o <= 0 for o in orders)):
                raise ValueError('Orders must be positive.')
        c = []
        for i in range(m):
            y1, y2 = (yi[i], yi[i + 1])
            if orders[i] is None:
                n1, n2 = (len(y1), len(y2))
            else:
                n = orders[i] + 1
                n1 = min(n // 2, len(y1))
                n2 = min(n - n1, len(y2))
                n1 = min(n - n2, len(y2))
                if n1 + n2 != n:
                    mesg = 'Point %g has %d derivatives, point %g has %d derivatives, but order %d requested' % (xi[i], len(y1), xi[i + 1], len(y2), orders[i])
                    raise ValueError(mesg)
                if not (n1 <= len(y1) and n2 <= len(y2)):
                    raise ValueError('`order` input incompatible with length y1 or y2.')
            b = BPoly._construct_from_derivatives(xi[i], xi[i + 1], y1[:n1], y2[:n2])
            if len(b) < k:
                b = BPoly._raise_degree(b, k - len(b))
            c.append(b)
        c = cupy.asarray(c)
        return cls(c.swapaxes(0, 1), xi, extrapolate)

    @staticmethod
    def _construct_from_derivatives(xa, xb, ya, yb):
        """
        Compute the coefficients of a polynomial in the Bernstein basis
        given the values and derivatives at the edges.

        Return the coefficients of a polynomial in the Bernstein basis
        defined on ``[xa, xb]`` and having the values and derivatives at the
        endpoints `xa` and `xb` as specified by `ya`` and `yb`.

        The polynomial constructed is of the minimal possible degree, i.e.,
        if the lengths of `ya` and `yb` are `na` and `nb`, the degree
        of the polynomial is ``na + nb - 1``.

        Parameters
        ----------
        xa : float
            Left-hand end point of the interval
        xb : float
            Right-hand end point of the interval
        ya : array_like
            Derivatives at `xa`. `ya[0]` is the value of the function, and
            `ya[i]` for ``i > 0`` is the value of the ``i``th derivative.
        yb : array_like
            Derivatives at `xb`.

        Returns
        -------
        array
            coefficient array of a polynomial having specified derivatives

        Notes
        -----
        This uses several facts from life of Bernstein basis functions.
        First of all,

            .. math:: b'_{a, n} = n (b_{a-1, n-1} - b_{a, n-1})

        If B(x) is a linear combination of the form

            .. math:: B(x) = \\sum_{a=0}^{n} c_a b_{a, n},

        then :math: B'(x) = n \\sum_{a=0}^{n-1} (c_{a+1} - c_{a}) b_{a, n-1}.
        Iterating the latter one, one finds for the q-th derivative

            .. math:: B^{q}(x) = n!/(n-q)! \\sum_{a=0}^{n-q} Q_a b_{a, n-q},

        with

            .. math:: Q_a = \\sum_{j=0}^{q} (-)^{j+q} comb(q, j) c_{j+a}

        This way, only `a=0` contributes to :math: `B^{q}(x = xa)`, and
        `c_q` are found one by one by iterating `q = 0, ..., na`.

        At ``x = xb`` it's the same with ``a = n - q``.
        """
        ya, yb = (cupy.asarray(ya), cupy.asarray(yb))
        if ya.shape[1:] != yb.shape[1:]:
            raise ValueError('Shapes of ya {} and yb {} are incompatible'.format(ya.shape, yb.shape))
        dta, dtb = (ya.dtype, yb.dtype)
        if cupy.issubdtype(dta, cupy.complexfloating) or cupy.issubdtype(dtb, cupy.complexfloating):
            dt = cupy.complex_
        else:
            dt = cupy.float_
        na, nb = (len(ya), len(yb))
        n = na + nb
        c = cupy.empty((na + nb,) + ya.shape[1:], dtype=dt)
        for q in range(0, na):
            c[q] = ya[q] / spec.poch(n - q, q) * (xb - xa) ** q
            for j in range(0, q):
                c[q] -= (-1) ** (j + q) * _comb(q, j) * c[j]
        for q in range(0, nb):
            c[-q - 1] = yb[q] / spec.poch(n - q, q) * (-1) ** q * (xb - xa) ** q
            for j in range(0, q):
                c[-q - 1] -= (-1) ** (j + 1) * _comb(q, j + 1) * c[-q + j]
        return c