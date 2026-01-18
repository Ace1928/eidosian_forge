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
class Exponentiation(Kernel):
    """The Exponentiation kernel takes one base kernel and a scalar parameter
    :math:`p` and combines them via

    .. math::
        k_{exp}(X, Y) = k(X, Y) ^p

    Note that the `__pow__` magic method is overridden, so
    `Exponentiation(RBF(), 2)` is equivalent to using the ** operator
    with `RBF() ** 2`.


    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    kernel : Kernel
        The base kernel

    exponent : float
        The exponent for the base kernel


    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import (RationalQuadratic,
    ...            Exponentiation)
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = Exponentiation(RationalQuadratic(), exponent=2)
    >>> gpr = GaussianProcessRegressor(kernel=kernel, alpha=5,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.419...
    >>> gpr.predict(X[:1,:], return_std=True)
    (array([635.5...]), array([0.559...]))
    """

    def __init__(self, kernel, exponent):
        self.kernel = kernel
        self.exponent = exponent

    def get_params(self, deep=True):
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = dict(kernel=self.kernel, exponent=self.exponent)
        if deep:
            deep_items = self.kernel.get_params().items()
            params.update((('kernel__' + k, val) for k, val in deep_items))
        return params

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter."""
        r = []
        for hyperparameter in self.kernel.hyperparameters:
            r.append(Hyperparameter('kernel__' + hyperparameter.name, hyperparameter.value_type, hyperparameter.bounds, hyperparameter.n_elements))
        return r

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        return self.kernel.theta

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        self.kernel.theta = theta

    @property
    def bounds(self):
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : ndarray of shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        return self.kernel.bounds

    def __eq__(self, b):
        if type(self) != type(b):
            return False
        return self.kernel == b.kernel and self.exponent == b.exponent

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_Y, n_features) or list of object,            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        if eval_gradient:
            K, K_gradient = self.kernel(X, Y, eval_gradient=True)
            K_gradient *= self.exponent * K[:, :, np.newaxis] ** (self.exponent - 1)
            return (K ** self.exponent, K_gradient)
        else:
            K = self.kernel(X, Y, eval_gradient=False)
            return K ** self.exponent

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return self.kernel.diag(X) ** self.exponent

    def __repr__(self):
        return '{0} ** {1}'.format(self.kernel, self.exponent)

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return self.kernel.is_stationary()

    @property
    def requires_vector_input(self):
        """Returns whether the kernel is defined on discrete structures."""
        return self.kernel.requires_vector_input