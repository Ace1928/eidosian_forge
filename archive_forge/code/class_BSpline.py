import operator
import cupy
from cupy._core import internal
from cupy._core._scalar import get_typename
from cupyx.scipy.sparse import csr_matrix
import numpy as np
class BSpline:
    """Univariate spline in the B-spline basis.

    .. math::
        S(x) = \\sum_{j=0}^{n-1} c_j  B_{j, k; t}(x)

    where :math:`B_{j, k; t}` are B-spline basis functions of degree `k`
    and knots `t`.

    Parameters
    ----------
    t : ndarray, shape (n+k+1,)
        knots
    c : ndarray, shape (>=n, ...)
        spline coefficients
    k : int
        B-spline degree
    extrapolate : bool or 'periodic', optional
        whether to extrapolate beyond the base interval, ``t[k] .. t[n]``,
        or to return nans.
        If True, extrapolates the first and last polynomial pieces of b-spline
        functions active on the base interval.
        If 'periodic', periodic extrapolation is used.
        Default is True.
    axis : int, optional
        Interpolation axis. Default is zero.

    Attributes
    ----------
    t : ndarray
        knot vector
    c : ndarray
        spline coefficients
    k : int
        spline degree
    extrapolate : bool
        If True, extrapolates the first and last polynomial pieces of b-spline
        functions active on the base interval.
    axis : int
        Interpolation axis.
    tck : tuple
        A read-only equivalent of ``(self.t, self.c, self.k)``

    Notes
    -----
    B-spline basis elements are defined via

    .. math::
        B_{i, 0}(x) = 1, \\textrm{if $t_i \\le x < t_{i+1}$, otherwise $0$,}

        B_{i, k}(x) = \\frac{x - t_i}{t_{i+k} - t_i} B_{i, k-1}(x)
                 + \\frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1, k-1}(x)

    **Implementation details**

    - At least ``k+1`` coefficients are required for a spline of degree `k`,
      so that ``n >= k+1``. Additional coefficients, ``c[j]`` with
      ``j > n``, are ignored.

    - B-spline basis elements of degree `k` form a partition of unity on the
      *base interval*, ``t[k] <= x <= t[n]``.

    - Based on [1]_ and [2]_

    .. seealso:: :class:`scipy.interpolate.BSpline`

    References
    ----------
    .. [1] Tom Lyche and Knut Morken, Spline methods,
        http://www.uio.no/studier/emner/matnat/ifi/INF-MAT5340/v05/undervisningsmateriale/
    .. [2] Carl de Boor, A practical guide to splines, Springer, 2001.
    """

    def __init__(self, t, c, k, extrapolate=True, axis=0):
        self.k = operator.index(k)
        self.c = cupy.asarray(c)
        self.t = cupy.ascontiguousarray(t, dtype=cupy.float64)
        if extrapolate == 'periodic':
            self.extrapolate = extrapolate
        else:
            self.extrapolate = bool(extrapolate)
        n = self.t.shape[0] - self.k - 1
        axis = internal._normalize_axis_index(axis, self.c.ndim)
        self.axis = axis
        if axis != 0:
            self.c = cupy.moveaxis(self.c, axis, 0)
        if k < 0:
            raise ValueError('Spline order cannot be negative.')
        if self.t.ndim != 1:
            raise ValueError('Knot vector must be one-dimensional.')
        if n < self.k + 1:
            raise ValueError('Need at least %d knots for degree %d' % (2 * k + 2, k))
        if (cupy.diff(self.t) < 0).any():
            raise ValueError('Knots must be in a non-decreasing order.')
        if len(cupy.unique(self.t[k:n + 1])) < 2:
            raise ValueError('Need at least two internal knots.')
        if not cupy.isfinite(self.t).all():
            raise ValueError('Knots should not have nans or infs.')
        if self.c.ndim < 1:
            raise ValueError('Coefficients must be at least 1-dimensional.')
        if self.c.shape[0] < n:
            raise ValueError('Knots, coefficients and degree are inconsistent.')
        dt = _get_dtype(self.c.dtype)
        self.c = cupy.ascontiguousarray(self.c, dtype=dt)

    @classmethod
    def construct_fast(cls, t, c, k, extrapolate=True, axis=0):
        """Construct a spline without making checks.
        Accepts same parameters as the regular constructor. Input arrays
        `t` and `c` must of correct shape and dtype.
        """
        self = object.__new__(cls)
        self.t, self.c, self.k = (t, c, k)
        self.extrapolate = extrapolate
        self.axis = axis
        return self

    @property
    def tck(self):
        """Equivalent to ``(self.t, self.c, self.k)`` (read-only).
        """
        return (self.t, self.c, self.k)

    @classmethod
    def basis_element(cls, t, extrapolate=True):
        """Return a B-spline basis element ``B(x | t[0], ..., t[k+1])``.

        Parameters
        ----------
        t : ndarray, shape (k+2,)
            internal knots
        extrapolate : bool or 'periodic', optional
            whether to extrapolate beyond the base interval,
            ``t[0] .. t[k+1]``, or to return nans.
            If 'periodic', periodic extrapolation is used.
            Default is True.

        Returns
        -------
        basis_element : callable
            A callable representing a B-spline basis element for the knot
            vector `t`.

        Notes
        -----
        The degree of the B-spline, `k`, is inferred from the length of `t` as
        ``len(t)-2``. The knot vector is constructed by appending and
        prepending ``k+1`` elements to internal knots `t`.

        .. seealso:: :class:`scipy.interpolate.BSpline`
        """
        k = len(t) - 2
        t = _as_float_array(t)
        t = cupy.r_[(t[0] - 1,) * k, t, (t[-1] + 1,) * k]
        c = cupy.zeros_like(t)
        c[k] = 1.0
        return cls.construct_fast(t, c, k, extrapolate)

    @classmethod
    def design_matrix(cls, x, t, k, extrapolate=False):
        """
        Returns a design matrix as a CSR format sparse array.

        Parameters
        ----------
        x : array_like, shape (n,)
            Points to evaluate the spline at.
        t : array_like, shape (nt,)
            Sorted 1D array of knots.
        k : int
            B-spline degree.
        extrapolate : bool or 'periodic', optional
            Whether to extrapolate based on the first and last intervals
            or raise an error. If 'periodic', periodic extrapolation is used.
            Default is False.

        Returns
        -------
        design_matrix : `csr_matrix` object
            Sparse matrix in CSR format where each row contains all the basis
            elements of the input row (first row = basis elements of x[0],
            ..., last row = basis elements x[-1]).

        Notes
        -----
        In each row of the design matrix all the basis elements are evaluated
        at the certain point (first row - x[0], ..., last row - x[-1]).
        `nt` is a length of the vector of knots: as far as there are
        `nt - k - 1` basis elements, `nt` should be not less than `2 * k + 2`
        to have at least `k + 1` basis element.

        Out of bounds `x` raises a ValueError.

        .. note::
            This method returns a `csr_matrix` instance as CuPy still does not
            have `csr_array`.

        .. seealso:: :class:`scipy.interpolate.BSpline`
        """
        x = _as_float_array(x, True)
        t = _as_float_array(t, True)
        if extrapolate != 'periodic':
            extrapolate = bool(extrapolate)
        if k < 0:
            raise ValueError('Spline order cannot be negative.')
        if t.ndim != 1 or np.any(t[1:] < t[:-1]):
            raise ValueError(f'Expect t to be a 1-D sorted array_like, but got t={t}.')
        if len(t) < 2 * k + 2:
            raise ValueError(f'Length t is not enough for k={k}.')
        if extrapolate == 'periodic':
            n = t.size - k - 1
            x = t[k] + (x - t[k]) % (t[n] - t[k])
            extrapolate = False
        elif not extrapolate and (min(x) < t[k] or max(x) > t[t.shape[0] - k - 1]):
            raise ValueError(f'Out of bounds w/ x = {x}.')
        n = x.shape[0]
        nnz = n * (k + 1)
        if nnz < cupy.iinfo(cupy.int32).max:
            int_dtype = cupy.int32
        else:
            int_dtype = cupy.int64
        indices = cupy.empty(n * (k + 1), dtype=int_dtype)
        indptr = cupy.arange(0, (n + 1) * (k + 1), k + 1, dtype=int_dtype)
        data, indices = _make_design_matrix(x, t, k, extrapolate, indices)
        return csr_matrix((data, indices, indptr), shape=(x.shape[0], t.shape[0] - k - 1))

    def __call__(self, x, nu=0, extrapolate=None):
        """
        Evaluate a spline function.

        Parameters
        ----------
        x : array_like
            points to evaluate the spline at.
        nu : int, optional
            derivative to evaluate (default is 0).
        extrapolate : bool or 'periodic', optional
            whether to extrapolate based on the first and last intervals
            or return nans. If 'periodic', periodic extrapolation is used.
            Default is `self.extrapolate`.

        Returns
        -------
        y : array_like
            Shape is determined by replacing the interpolation axis
            in the coefficient array with the shape of `x`.
        """
        if extrapolate is None:
            extrapolate = self.extrapolate
        x = cupy.asarray(x)
        x_shape, x_ndim = (x.shape, x.ndim)
        x = cupy.ascontiguousarray(cupy.ravel(x), dtype=cupy.float_)
        if extrapolate == 'periodic':
            n = self.t.size - self.k - 1
            x = self.t[self.k] + (x - self.t[self.k]) % (self.t[n] - self.t[self.k])
            extrapolate = False
        out = cupy.empty((len(x), int(np.prod(self.c.shape[1:]))), dtype=self.c.dtype)
        self._evaluate(x, nu, extrapolate, out)
        out = out.reshape(x_shape + self.c.shape[1:])
        if self.axis != 0:
            dim_order = list(range(out.ndim))
            dim_order = dim_order[x_ndim:x_ndim + self.axis] + dim_order[:x_ndim] + dim_order[x_ndim + self.axis:]
            out = out.transpose(dim_order)
        return out

    def _ensure_c_contiguous(self):
        if not self.t.flags.c_contiguous:
            self.t = self.t.copy()
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()

    def _evaluate(self, xp, nu, extrapolate, out):
        _evaluate_spline(self.t, self.c.reshape(self.c.shape[0], -1), self.k, xp, nu, extrapolate, out)

    def derivative(self, nu=1):
        """
        Return a B-spline representing the derivative.

        Parameters
        ----------
        nu : int, optional
            Derivative order.
            Default is 1.

        Returns
        -------
        b : BSpline object
            A new instance representing the derivative.

        See Also
        --------
        splder, splantider
        """
        c = self.c
        ct = len(self.t) - len(c)
        if ct > 0:
            c = cupy.r_[c, cupy.zeros((ct,) + c.shape[1:])]
        tck = splder((self.t, c, self.k), nu)
        return self.construct_fast(*tck, extrapolate=self.extrapolate, axis=self.axis)

    def antiderivative(self, nu=1):
        """
        Return a B-spline representing the antiderivative.

        Parameters
        ----------
        nu : int, optional
            Antiderivative order. Default is 1.

        Returns
        -------
        b : BSpline object
            A new instance representing the antiderivative.

        Notes
        -----
        If antiderivative is computed and ``self.extrapolate='periodic'``,
        it will be set to False for the returned instance. This is done because
        the antiderivative is no longer periodic and its correct evaluation
        outside of the initially given x interval is difficult.

        See Also
        --------
        splder, splantider
        """
        c = self.c
        ct = len(self.t) - len(c)
        if ct > 0:
            c = cupy.r_[c, cupy.zeros((ct,) + c.shape[1:])]
        tck = splantider((self.t, c, self.k), nu)
        if self.extrapolate == 'periodic':
            extrapolate = False
        else:
            extrapolate = self.extrapolate
        return self.construct_fast(*tck, extrapolate=extrapolate, axis=self.axis)

    def integrate(self, a, b, extrapolate=None):
        """
        Compute a definite integral of the spline.

        Parameters
        ----------
        a : float
            Lower limit of integration.
        b : float
            Upper limit of integration.
        extrapolate : bool or 'periodic', optional
            whether to extrapolate beyond the base interval,
            ``t[k] .. t[-k-1]``, or take the spline to be zero outside of the
            base interval. If 'periodic', periodic extrapolation is used.
            If None (default), use `self.extrapolate`.

        Returns
        -------
        I : array_like
            Definite integral of the spline over the interval ``[a, b]``.
        """
        if extrapolate is None:
            extrapolate = self.extrapolate
        self._ensure_c_contiguous()
        sign = 1
        if b < a:
            a, b = (b, a)
            sign = -1
        n = self.t.size - self.k - 1
        if extrapolate != 'periodic' and (not extrapolate):
            a = max(a, self.t[self.k].item())
            b = min(b, self.t[n].item())
        out = cupy.empty((2, int(np.prod(self.c.shape[1:]))), dtype=self.c.dtype)
        c = self.c
        ct = len(self.t) - len(c)
        if ct > 0:
            c = cupy.r_[c, cupy.zeros((ct,) + c.shape[1:])]
        ta, ca, ka = splantider((self.t, c, self.k), 1)
        if extrapolate == 'periodic':
            ts, te = (self.t[self.k], self.t[n])
            period = te - ts
            interval = b - a
            n_periods, left = divmod(interval, period)
            if n_periods > 0:
                x = cupy.asarray([ts, te], dtype=cupy.float_)
                _evaluate_spline(ta, ca.reshape(ca.shape[0], -1), ka, x, 0, False, out)
                integral = out[1] - out[0]
                integral *= n_periods
            else:
                integral = cupy.zeros((1, int(np.prod(self.c.shape[1:]))), dtype=self.c.dtype)
            a = ts + (a - ts) % period
            b = a + left
            if b <= te:
                x = cupy.asarray([a, b], dtype=cupy.float_)
                _evaluate_spline(ta, ca.reshape(ca.shape[0], -1), ka, x, 0, False, out)
                integral += out[1] - out[0]
            else:
                x = cupy.asarray([a, te], dtype=cupy.float_)
                _evaluate_spline(ta, ca.reshape(ca.shape[0], -1), ka, x, 0, False, out)
                integral += out[1] - out[0]
                x = cupy.asarray([ts, ts + b - te], dtype=cupy.float_)
                _evaluate_spline(ta, ca.reshape(ca.shape[0], -1), ka, x, 0, False, out)
                integral += out[1] - out[0]
        else:
            x = cupy.asarray([a, b], dtype=cupy.float_)
            _evaluate_spline(ta, ca.reshape(ca.shape[0], -1), ka, x, 0, extrapolate, out)
            integral = out[1] - out[0]
        integral *= sign
        return integral.reshape(ca.shape[1:])