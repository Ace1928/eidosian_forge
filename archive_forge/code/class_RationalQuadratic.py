import math
import warnings
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from inspect import signature
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import gamma, kv
from ..base import clone
from ..exceptions import ConvergenceWarning
from ..metrics.pairwise import pairwise_kernels
from ..utils.validation import _num_samples
class RationalQuadratic(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Rational Quadratic kernel.

    The RationalQuadratic kernel can be seen as a scale mixture (an infinite
    sum) of RBF kernels with different characteristic length scales. It is
    parameterized by a length scale parameter :math:`l>0` and a scale
    mixture parameter :math:`\\alpha>0`. Only the isotropic variant
    where length_scale :math:`l` is a scalar is supported at the moment.
    The kernel is given by:

    .. math::
        k(x_i, x_j) = \\left(
        1 + \\frac{d(x_i, x_j)^2 }{ 2\\alpha  l^2}\\right)^{-\\alpha}

    where :math:`\\alpha` is the scale mixture parameter, :math:`l` is
    the length scale of the kernel and :math:`d(\\cdot,\\cdot)` is the
    Euclidean distance.
    For advice on how to set the parameters, see e.g. [1]_.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    length_scale : float > 0, default=1.0
        The length scale of the kernel.

    alpha : float > 0, default=1.0
        Scale mixture parameter

    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.

    alpha_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'alpha'.
        If set to "fixed", 'alpha' cannot be changed during
        hyperparameter tuning.

    References
    ----------
    .. [1] `David Duvenaud (2014). "The Kernel Cookbook:
        Advice on Covariance functions".
        <https://www.cs.toronto.edu/~duvenaud/cookbook/>`_

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import RationalQuadratic
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = RationalQuadratic(length_scale=1.0, alpha=1.5)
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9733...
    >>> gpc.predict_proba(X[:2,:])
    array([[0.8881..., 0.0566..., 0.05518...],
            [0.8678..., 0.0707... , 0.0614...]])
    """

    def __init__(self, length_scale=1.0, alpha=1.0, length_scale_bounds=(1e-05, 100000.0), alpha_bounds=(1e-05, 100000.0)):
        self.length_scale = length_scale
        self.alpha = alpha
        self.length_scale_bounds = length_scale_bounds
        self.alpha_bounds = alpha_bounds

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter('length_scale', 'numeric', self.length_scale_bounds)

    @property
    def hyperparameter_alpha(self):
        return Hyperparameter('alpha', 'numeric', self.alpha_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if len(np.atleast_1d(self.length_scale)) > 1:
            raise AttributeError('RationalQuadratic kernel only supports isotropic version, please use a single scalar for length_scale')
        X = np.atleast_2d(X)
        if Y is None:
            dists = squareform(pdist(X, metric='sqeuclidean'))
            tmp = dists / (2 * self.alpha * self.length_scale ** 2)
            base = 1 + tmp
            K = base ** (-self.alpha)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError('Gradient can only be evaluated when Y is None.')
            dists = cdist(X, Y, metric='sqeuclidean')
            K = (1 + dists / (2 * self.alpha * self.length_scale ** 2)) ** (-self.alpha)
        if eval_gradient:
            if not self.hyperparameter_length_scale.fixed:
                length_scale_gradient = dists * K / (self.length_scale ** 2 * base)
                length_scale_gradient = length_scale_gradient[:, :, np.newaxis]
            else:
                length_scale_gradient = np.empty((K.shape[0], K.shape[1], 0))
            if not self.hyperparameter_alpha.fixed:
                alpha_gradient = K * (-self.alpha * np.log(base) + dists / (2 * self.length_scale ** 2 * base))
                alpha_gradient = alpha_gradient[:, :, np.newaxis]
            else:
                alpha_gradient = np.empty((K.shape[0], K.shape[1], 0))
            return (K, np.dstack((alpha_gradient, length_scale_gradient)))
        else:
            return K

    def __repr__(self):
        return '{0}(alpha={1:.3g}, length_scale={2:.3g})'.format(self.__class__.__name__, self.alpha, self.length_scale)