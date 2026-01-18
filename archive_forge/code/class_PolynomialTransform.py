import math
import textwrap
from abc import ABC, abstractmethod
import numpy as np
from scipy import spatial
from .._shared.utils import safe_as_int
from .._shared.compat import NP_COPY_IF_NEEDED
class PolynomialTransform(_GeometricTransform):
    """2D polynomial transformation.

    Has the following form::

        X = sum[j=0:order]( sum[i=0:j]( a_ji * x**(j - i) * y**i ))
        Y = sum[j=0:order]( sum[i=0:j]( b_ji * x**(j - i) * y**i ))

    Parameters
    ----------
    params : (2, N) array_like, optional
        Polynomial coefficients where `N * 2 = (order + 1) * (order + 2)`. So,
        a_ji is defined in `params[0, :]` and b_ji in `params[1, :]`.

    Attributes
    ----------
    params : (2, N) array
        Polynomial coefficients where `N * 2 = (order + 1) * (order + 2)`. So,
        a_ji is defined in `params[0, :]` and b_ji in `params[1, :]`.

    """

    def __init__(self, params=None, *, dimensionality=2):
        if dimensionality != 2:
            raise NotImplementedError('Polynomial transforms are only implemented for 2D.')
        if params is None:
            params = np.array([[0, 1, 0], [0, 0, 1]])
        else:
            params = np.asarray(params)
        if params.shape[0] != 2:
            raise ValueError('invalid shape of transformation parameters')
        self.params = params

    def estimate(self, src, dst, order=2, weights=None):
        """Estimate the transformation from a set of corresponding points.

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        The transformation is defined as::

            X = sum[j=0:order]( sum[i=0:j]( a_ji * x**(j - i) * y**i ))
            Y = sum[j=0:order]( sum[i=0:j]( b_ji * x**(j - i) * y**i ))

        These equations can be transformed to the following form::

            0 = sum[j=0:order]( sum[i=0:j]( a_ji * x**(j - i) * y**i )) - X
            0 = sum[j=0:order]( sum[i=0:j]( b_ji * x**(j - i) * y**i )) - Y

        which exist for each set of corresponding points, so we have a set of
        N * 2 equations. The coefficients appear linearly so we can write
        A x = 0, where::

            A   = [[1 x y x**2 x*y y**2 ... 0 ...             0 -X]
                   [0 ...                 0 1 x y x**2 x*y y**2 -Y]
                    ...
                    ...
                  ]
            x.T = [a00 a10 a11 a20 a21 a22 ... ann
                   b00 b10 b11 b20 b21 b22 ... bnn c3]

        In case of total least-squares the solution of this homogeneous system
        of equations is the right singular vector of A which corresponds to the
        smallest singular value normed by the coefficient c3.

        Weights can be applied to each pair of corresponding points to
        indicate, particularly in an overdetermined system, if point pairs have
        higher or lower confidence or uncertainties associated with them. From
        the matrix treatment of least squares problems, these weight values are
        normalised, square-rooted, then built into a diagonal matrix, by which
        A is multiplied.

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.
        order : int, optional
            Polynomial order (number of coefficients is order + 1).
        weights : (N,) array_like, optional
            Relative weight values for each pair of points.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """
        src = np.asarray(src)
        dst = np.asarray(dst)
        xs = src[:, 0]
        ys = src[:, 1]
        xd = dst[:, 0]
        yd = dst[:, 1]
        rows = src.shape[0]
        order = safe_as_int(order)
        u = (order + 1) * (order + 2)
        A = np.zeros((rows * 2, u + 1))
        pidx = 0
        for j in range(order + 1):
            for i in range(j + 1):
                A[:rows, pidx] = xs ** (j - i) * ys ** i
                A[rows:, pidx + u // 2] = xs ** (j - i) * ys ** i
                pidx += 1
        A[:rows, -1] = xd
        A[rows:, -1] = yd
        if weights is None:
            _, _, V = np.linalg.svd(A)
        else:
            weights = np.asarray(weights)
            W = np.diag(np.tile(np.sqrt(weights / np.max(weights)), 2))
            _, _, V = np.linalg.svd(W @ A)
        params = -V[-1, :-1] / V[-1, -1]
        self.params = params.reshape((2, u // 2))
        return True

    def __call__(self, coords):
        """Apply forward transformation.

        Parameters
        ----------
        coords : (N, 2) array_like
            source coordinates

        Returns
        -------
        coords : (N, 2) array
            Transformed coordinates.

        """
        coords = np.asarray(coords)
        x = coords[:, 0]
        y = coords[:, 1]
        u = len(self.params.ravel())
        order = int((-3 + math.sqrt(9 - 4 * (2 - u))) / 2)
        dst = np.zeros(coords.shape)
        pidx = 0
        for j in range(order + 1):
            for i in range(j + 1):
                dst[:, 0] += self.params[0, pidx] * x ** (j - i) * y ** i
                dst[:, 1] += self.params[1, pidx] * x ** (j - i) * y ** i
                pidx += 1
        return dst

    @property
    def inverse(self):
        raise NotImplementedError('There is no explicit way to do the inverse polynomial transformation. Instead, estimate the inverse transformation parameters by exchanging source and destination coordinates,then apply the forward transformation.')