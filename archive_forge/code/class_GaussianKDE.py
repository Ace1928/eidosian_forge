import functools
from numbers import Number
import numpy as np
from matplotlib import _api, _docstring, cbook
class GaussianKDE:
    """
    Representation of a kernel-density estimate using Gaussian kernels.

    Parameters
    ----------
    dataset : array-like
        Datapoints to estimate from. In case of univariate data this is a 1-D
        array, otherwise a 2D array with shape (# of dims, # of data).
    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth.  This can be
        'scott', 'silverman', a scalar constant or a callable.  If a
        scalar, this will be used directly as `kde.factor`.  If a
        callable, it should take a `GaussianKDE` instance as only
        parameter and return a scalar. If None (default), 'scott' is used.

    Attributes
    ----------
    dataset : ndarray
        The dataset passed to the constructor.
    dim : int
        Number of dimensions.
    num_dp : int
        Number of datapoints.
    factor : float
        The bandwidth factor, obtained from `kde.covariance_factor`, with which
        the covariance matrix is multiplied.
    covariance : ndarray
        The covariance matrix of *dataset*, scaled by the calculated bandwidth
        (`kde.factor`).
    inv_cov : ndarray
        The inverse of *covariance*.

    Methods
    -------
    kde.evaluate(points) : ndarray
        Evaluate the estimated pdf on a provided set of points.
    kde(points) : ndarray
        Same as kde.evaluate(points)
    """

    def __init__(self, dataset, bw_method=None):
        self.dataset = np.atleast_2d(dataset)
        if not np.array(self.dataset).size > 1:
            raise ValueError('`dataset` input should have multiple elements.')
        self.dim, self.num_dp = np.array(self.dataset).shape
        if bw_method is None:
            pass
        elif cbook._str_equal(bw_method, 'scott'):
            self.covariance_factor = self.scotts_factor
        elif cbook._str_equal(bw_method, 'silverman'):
            self.covariance_factor = self.silverman_factor
        elif isinstance(bw_method, Number):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            raise ValueError("`bw_method` should be 'scott', 'silverman', a scalar or a callable")
        self.factor = self.covariance_factor()
        if not hasattr(self, '_data_inv_cov'):
            self.data_covariance = np.atleast_2d(np.cov(self.dataset, rowvar=1, bias=False))
            self.data_inv_cov = np.linalg.inv(self.data_covariance)
        self.covariance = self.data_covariance * self.factor ** 2
        self.inv_cov = self.data_inv_cov / self.factor ** 2
        self.norm_factor = np.sqrt(np.linalg.det(2 * np.pi * self.covariance)) * self.num_dp

    def scotts_factor(self):
        return np.power(self.num_dp, -1.0 / (self.dim + 4))

    def silverman_factor(self):
        return np.power(self.num_dp * (self.dim + 2.0) / 4.0, -1.0 / (self.dim + 4))
    covariance_factor = scotts_factor

    def evaluate(self, points):
        """
        Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError : if the dimensionality of the input points is different
                     than the dimensionality of the KDE.

        """
        points = np.atleast_2d(points)
        dim, num_m = np.array(points).shape
        if dim != self.dim:
            raise ValueError(f'points have dimension {dim}, dataset has dimension {self.dim}')
        result = np.zeros(num_m)
        if num_m >= self.num_dp:
            for i in range(self.num_dp):
                diff = self.dataset[:, i, np.newaxis] - points
                tdiff = np.dot(self.inv_cov, diff)
                energy = np.sum(diff * tdiff, axis=0) / 2.0
                result = result + np.exp(-energy)
        else:
            for i in range(num_m):
                diff = self.dataset - points[:, i, np.newaxis]
                tdiff = np.dot(self.inv_cov, diff)
                energy = np.sum(diff * tdiff, axis=0) / 2.0
                result[i] = np.sum(np.exp(-energy), axis=0)
        result = result / self.norm_factor
        return result
    __call__ = evaluate