from math import prod
import warnings
import numpy as np
from numpy import (array, transpose, searchsorted, atleast_1d, atleast_2d,
import scipy.special as spec
from scipy.special import comb
from . import _fitpack_py
from . import dfitpack
from ._polyint import _Interpolator1D
from . import _ppoly
from .interpnd import _ndim_coords_from_arrays
from ._bsplines import make_interp_spline, BSpline
import itertools  # noqa: F401
class NdPPoly:
    """
    Piecewise tensor product polynomial

    The value at point ``xp = (x', y', z', ...)`` is evaluated by first
    computing the interval indices `i` such that::

        x[0][i[0]] <= x' < x[0][i[0]+1]
        x[1][i[1]] <= y' < x[1][i[1]+1]
        ...

    and then computing::

        S = sum(c[k0-m0-1,...,kn-mn-1,i[0],...,i[n]]
                * (xp[0] - x[0][i[0]])**m0
                * ...
                * (xp[n] - x[n][i[n]])**mn
                for m0 in range(k[0]+1)
                ...
                for mn in range(k[n]+1))

    where ``k[j]`` is the degree of the polynomial in dimension j. This
    representation is the piecewise multivariate power basis.

    Parameters
    ----------
    c : ndarray, shape (k0, ..., kn, m0, ..., mn, ...)
        Polynomial coefficients, with polynomial order `kj` and
        `mj+1` intervals for each dimension `j`.
    x : ndim-tuple of ndarrays, shapes (mj+1,)
        Polynomial breakpoints for each dimension. These must be
        sorted in increasing order.
    extrapolate : bool, optional
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs. Default: True.

    Attributes
    ----------
    x : tuple of ndarrays
        Breakpoints.
    c : ndarray
        Coefficients of the polynomials.

    Methods
    -------
    __call__
    derivative
    antiderivative
    integrate
    integrate_1d
    construct_fast

    See also
    --------
    PPoly : piecewise polynomials in 1D

    Notes
    -----
    High-order polynomials in the power basis can be numerically
    unstable.

    """

    def __init__(self, c, x, extrapolate=None):
        self.x = tuple((np.ascontiguousarray(v, dtype=np.float64) for v in x))
        self.c = np.asarray(c)
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = bool(extrapolate)
        ndim = len(self.x)
        if any((v.ndim != 1 for v in self.x)):
            raise ValueError('x arrays must all be 1-dimensional')
        if any((v.size < 2 for v in self.x)):
            raise ValueError('x arrays must all contain at least 2 points')
        if c.ndim < 2 * ndim:
            raise ValueError('c must have at least 2*len(x) dimensions')
        if any((np.any(v[1:] - v[:-1] < 0) for v in self.x)):
            raise ValueError('x-coordinates are not in increasing order')
        if any((a != b.size - 1 for a, b in zip(c.shape[ndim:2 * ndim], self.x))):
            raise ValueError('x and c do not agree on the number of intervals')
        dtype = self._get_dtype(self.c.dtype)
        self.c = np.ascontiguousarray(self.c, dtype=dtype)

    @classmethod
    def construct_fast(cls, c, x, extrapolate=None):
        """
        Construct the piecewise polynomial without making checks.

        Takes the same parameters as the constructor. Input arguments
        ``c`` and ``x`` must be arrays of the correct shape and type.  The
        ``c`` array can only be of dtypes float and complex, and ``x``
        array must have dtype float.

        """
        self = object.__new__(cls)
        self.c = c
        self.x = x
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = extrapolate
        return self

    def _get_dtype(self, dtype):
        if np.issubdtype(dtype, np.complexfloating) or np.issubdtype(self.c.dtype, np.complexfloating):
            return np.complex128
        else:
            return np.float64

    def _ensure_c_contiguous(self):
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()
        if not isinstance(self.x, tuple):
            self.x = tuple(self.x)

    def __call__(self, x, nu=None, extrapolate=None):
        """
        Evaluate the piecewise polynomial or its derivative

        Parameters
        ----------
        x : array-like
            Points to evaluate the interpolant at.
        nu : tuple, optional
            Orders of derivatives to evaluate. Each must be non-negative.
        extrapolate : bool, optional
            Whether to extrapolate to out-of-bounds points based on first
            and last intervals, or to return NaNs.

        Returns
        -------
        y : array-like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.

        """
        if extrapolate is None:
            extrapolate = self.extrapolate
        else:
            extrapolate = bool(extrapolate)
        ndim = len(self.x)
        x = _ndim_coords_from_arrays(x)
        x_shape = x.shape
        x = np.ascontiguousarray(x.reshape(-1, x.shape[-1]), dtype=np.float64)
        if nu is None:
            nu = np.zeros((ndim,), dtype=np.intc)
        else:
            nu = np.asarray(nu, dtype=np.intc)
            if nu.ndim != 1 or nu.shape[0] != ndim:
                raise ValueError('invalid number of derivative orders nu')
        dim1 = prod(self.c.shape[:ndim])
        dim2 = prod(self.c.shape[ndim:2 * ndim])
        dim3 = prod(self.c.shape[2 * ndim:])
        ks = np.array(self.c.shape[:ndim], dtype=np.intc)
        out = np.empty((x.shape[0], dim3), dtype=self.c.dtype)
        self._ensure_c_contiguous()
        _ppoly.evaluate_nd(self.c.reshape(dim1, dim2, dim3), self.x, ks, x, nu, bool(extrapolate), out)
        return out.reshape(x_shape[:-1] + self.c.shape[2 * ndim:])

    def _derivative_inplace(self, nu, axis):
        """
        Compute 1-D derivative along a selected dimension in-place
        May result to non-contiguous c array.
        """
        if nu < 0:
            return self._antiderivative_inplace(-nu, axis)
        ndim = len(self.x)
        axis = axis % ndim
        if nu == 0:
            return
        else:
            sl = [slice(None)] * ndim
            sl[axis] = slice(None, -nu, None)
            c2 = self.c[tuple(sl)]
        if c2.shape[axis] == 0:
            shp = list(c2.shape)
            shp[axis] = 1
            c2 = np.zeros(shp, dtype=c2.dtype)
        factor = spec.poch(np.arange(c2.shape[axis], 0, -1), nu)
        sl = [None] * c2.ndim
        sl[axis] = slice(None)
        c2 *= factor[tuple(sl)]
        self.c = c2

    def _antiderivative_inplace(self, nu, axis):
        """
        Compute 1-D antiderivative along a selected dimension
        May result to non-contiguous c array.
        """
        if nu <= 0:
            return self._derivative_inplace(-nu, axis)
        ndim = len(self.x)
        axis = axis % ndim
        perm = list(range(ndim))
        perm[0], perm[axis] = (perm[axis], perm[0])
        perm = perm + list(range(ndim, self.c.ndim))
        c = self.c.transpose(perm)
        c2 = np.zeros((c.shape[0] + nu,) + c.shape[1:], dtype=c.dtype)
        c2[:-nu] = c
        factor = spec.poch(np.arange(c.shape[0], 0, -1), nu)
        c2[:-nu] /= factor[(slice(None),) + (None,) * (c.ndim - 1)]
        perm2 = list(range(c2.ndim))
        perm2[1], perm2[ndim + axis] = (perm2[ndim + axis], perm2[1])
        c2 = c2.transpose(perm2)
        c2 = c2.copy()
        _ppoly.fix_continuity(c2.reshape(c2.shape[0], c2.shape[1], -1), self.x[axis], nu - 1)
        c2 = c2.transpose(perm2)
        c2 = c2.transpose(perm)
        self.c = c2

    def derivative(self, nu):
        """
        Construct a new piecewise polynomial representing the derivative.

        Parameters
        ----------
        nu : ndim-tuple of int
            Order of derivatives to evaluate for each dimension.
            If negative, the antiderivative is returned.

        Returns
        -------
        pp : NdPPoly
            Piecewise polynomial of orders (k[0] - nu[0], ..., k[n] - nu[n])
            representing the derivative of this polynomial.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals in each dimension are
        considered half-open, ``[a, b)``, except for the last interval
        which is closed ``[a, b]``.

        """
        p = self.construct_fast(self.c.copy(), self.x, self.extrapolate)
        for axis, n in enumerate(nu):
            p._derivative_inplace(n, axis)
        p._ensure_c_contiguous()
        return p

    def antiderivative(self, nu):
        """
        Construct a new piecewise polynomial representing the antiderivative.

        Antiderivative is also the indefinite integral of the function,
        and derivative is its inverse operation.

        Parameters
        ----------
        nu : ndim-tuple of int
            Order of derivatives to evaluate for each dimension.
            If negative, the derivative is returned.

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

        """
        p = self.construct_fast(self.c.copy(), self.x, self.extrapolate)
        for axis, n in enumerate(nu):
            p._antiderivative_inplace(n, axis)
        p._ensure_c_contiguous()
        return p

    def integrate_1d(self, a, b, axis, extrapolate=None):
        """
        Compute NdPPoly representation for one dimensional definite integral

        The result is a piecewise polynomial representing the integral:

        .. math::

           p(y, z, ...) = \\int_a^b dx\\, p(x, y, z, ...)

        where the dimension integrated over is specified with the
        `axis` parameter.

        Parameters
        ----------
        a, b : float
            Lower and upper bound for integration.
        axis : int
            Dimension over which to compute the 1-D integrals
        extrapolate : bool, optional
            Whether to extrapolate to out-of-bounds points based on first
            and last intervals, or to return NaNs.

        Returns
        -------
        ig : NdPPoly or array-like
            Definite integral of the piecewise polynomial over [a, b].
            If the polynomial was 1D, an array is returned,
            otherwise, an NdPPoly object.

        """
        if extrapolate is None:
            extrapolate = self.extrapolate
        else:
            extrapolate = bool(extrapolate)
        ndim = len(self.x)
        axis = int(axis) % ndim
        c = self.c
        swap = list(range(c.ndim))
        swap.insert(0, swap[axis])
        del swap[axis + 1]
        swap.insert(1, swap[ndim + axis])
        del swap[ndim + axis + 1]
        c = c.transpose(swap)
        p = PPoly.construct_fast(c.reshape(c.shape[0], c.shape[1], -1), self.x[axis], extrapolate=extrapolate)
        out = p.integrate(a, b, extrapolate=extrapolate)
        if ndim == 1:
            return out.reshape(c.shape[2:])
        else:
            c = out.reshape(c.shape[2:])
            x = self.x[:axis] + self.x[axis + 1:]
            return self.construct_fast(c, x, extrapolate=extrapolate)

    def integrate(self, ranges, extrapolate=None):
        """
        Compute a definite integral over a piecewise polynomial.

        Parameters
        ----------
        ranges : ndim-tuple of 2-tuples float
            Sequence of lower and upper bounds for each dimension,
            ``[(a[0], b[0]), ..., (a[ndim-1], b[ndim-1])]``
        extrapolate : bool, optional
            Whether to extrapolate to out-of-bounds points based on first
            and last intervals, or to return NaNs.

        Returns
        -------
        ig : array_like
            Definite integral of the piecewise polynomial over
            [a[0], b[0]] x ... x [a[ndim-1], b[ndim-1]]

        """
        ndim = len(self.x)
        if extrapolate is None:
            extrapolate = self.extrapolate
        else:
            extrapolate = bool(extrapolate)
        if not hasattr(ranges, '__len__') or len(ranges) != ndim:
            raise ValueError('Range not a sequence of correct length')
        self._ensure_c_contiguous()
        c = self.c
        for n, (a, b) in enumerate(ranges):
            swap = list(range(c.ndim))
            swap.insert(1, swap[ndim - n])
            del swap[ndim - n + 1]
            c = c.transpose(swap)
            p = PPoly.construct_fast(c, self.x[n], extrapolate=extrapolate)
            out = p.integrate(a, b, extrapolate=extrapolate)
            c = out.reshape(c.shape[2:])
        return c