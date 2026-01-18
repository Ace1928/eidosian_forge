from functools import lru_cache
import math
import warnings
import numpy as np
from matplotlib import _api
class BezierSegment:
    """
    A d-dimensional Bézier segment.

    Parameters
    ----------
    control_points : (N, d) array
        Location of the *N* control points.
    """

    def __init__(self, control_points):
        self._cpoints = np.asarray(control_points)
        self._N, self._d = self._cpoints.shape
        self._orders = np.arange(self._N)
        coeff = [math.factorial(self._N - 1) // (math.factorial(i) * math.factorial(self._N - 1 - i)) for i in range(self._N)]
        self._px = (self._cpoints.T * coeff).T

    def __call__(self, t):
        """
        Evaluate the Bézier curve at point(s) *t* in [0, 1].

        Parameters
        ----------
        t : (k,) array-like
            Points at which to evaluate the curve.

        Returns
        -------
        (k, d) array
            Value of the curve for each point in *t*.
        """
        t = np.asarray(t)
        return np.power.outer(1 - t, self._orders[::-1]) * np.power.outer(t, self._orders) @ self._px

    def point_at_t(self, t):
        """
        Evaluate the curve at a single point, returning a tuple of *d* floats.
        """
        return tuple(self(t))

    @property
    def control_points(self):
        """The control points of the curve."""
        return self._cpoints

    @property
    def dimension(self):
        """The dimension of the curve."""
        return self._d

    @property
    def degree(self):
        """Degree of the polynomial. One less the number of control points."""
        return self._N - 1

    @property
    def polynomial_coefficients(self):
        """
        The polynomial coefficients of the Bézier curve.

        .. warning:: Follows opposite convention from `numpy.polyval`.

        Returns
        -------
        (n+1, d) array
            Coefficients after expanding in polynomial basis, where :math:`n`
            is the degree of the Bézier curve and :math:`d` its dimension.
            These are the numbers (:math:`C_j`) such that the curve can be
            written :math:`\\sum_{j=0}^n C_j t^j`.

        Notes
        -----
        The coefficients are calculated as

        .. math::

            {n \\choose j} \\sum_{i=0}^j (-1)^{i+j} {j \\choose i} P_i

        where :math:`P_i` are the control points of the curve.
        """
        n = self.degree
        if n > 10:
            warnings.warn('Polynomial coefficients formula unstable for high order Bezier curves!', RuntimeWarning)
        P = self.control_points
        j = np.arange(n + 1)[:, None]
        i = np.arange(n + 1)[None, :]
        prefactor = (-1) ** (i + j) * _comb(j, i)
        return _comb(n, j) * prefactor @ P

    def axis_aligned_extrema(self):
        """
        Return the dimension and location of the curve's interior extrema.

        The extrema are the points along the curve where one of its partial
        derivatives is zero.

        Returns
        -------
        dims : array of int
            Index :math:`i` of the partial derivative which is zero at each
            interior extrema.
        dzeros : array of float
            Of same size as dims. The :math:`t` such that :math:`d/dx_i B(t) =
            0`
        """
        n = self.degree
        if n <= 1:
            return (np.array([]), np.array([]))
        Cj = self.polynomial_coefficients
        dCj = np.arange(1, n + 1)[:, None] * Cj[1:]
        dims = []
        roots = []
        for i, pi in enumerate(dCj.T):
            r = np.roots(pi[::-1])
            roots.append(r)
            dims.append(np.full_like(r, i))
        roots = np.concatenate(roots)
        dims = np.concatenate(dims)
        in_range = np.isreal(roots) & (roots >= 0) & (roots <= 1)
        return (dims[in_range], np.real(roots)[in_range])