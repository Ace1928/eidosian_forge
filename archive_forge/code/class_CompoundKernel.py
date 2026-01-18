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
class CompoundKernel(Kernel):
    """Kernel which is composed of a set of other kernels.

    .. versionadded:: 0.18

    Parameters
    ----------
    kernels : list of Kernels
        The other kernels

    Examples
    --------
    >>> from sklearn.gaussian_process.kernels import WhiteKernel
    >>> from sklearn.gaussian_process.kernels import RBF
    >>> from sklearn.gaussian_process.kernels import CompoundKernel
    >>> kernel = CompoundKernel(
    ...     [WhiteKernel(noise_level=3.0), RBF(length_scale=2.0)])
    >>> print(kernel.bounds)
    [[-11.51292546  11.51292546]
     [-11.51292546  11.51292546]]
    >>> print(kernel.n_dims)
    2
    >>> print(kernel.theta)
    [1.09861229 0.69314718]
    """

    def __init__(self, kernels):
        self.kernels = kernels

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
        return dict(kernels=self.kernels)

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
        return np.hstack([kernel.theta for kernel in self.kernels])

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : array of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        k_dims = self.k1.n_dims
        for i, kernel in enumerate(self.kernels):
            kernel.theta = theta[i * k_dims:(i + 1) * k_dims]

    @property
    def bounds(self):
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : array of shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        return np.vstack([kernel.bounds for kernel in self.kernels])

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Note that this compound kernel returns the results of all simple kernel
        stacked along an additional axis.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object,             default=None
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_X, n_features) or list of object,             default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of the
            kernel hyperparameter is computed.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y, n_kernels)
            Kernel k(X, Y)

        K_gradient : ndarray of shape                 (n_samples_X, n_samples_X, n_dims, n_kernels), optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        if eval_gradient:
            K = []
            K_grad = []
            for kernel in self.kernels:
                K_single, K_grad_single = kernel(X, Y, eval_gradient)
                K.append(K_single)
                K_grad.append(K_grad_single[..., np.newaxis])
            return (np.dstack(K), np.concatenate(K_grad, 3))
        else:
            return np.dstack([kernel(X, Y, eval_gradient) for kernel in self.kernels])

    def __eq__(self, b):
        if type(self) != type(b) or len(self.kernels) != len(b.kernels):
            return False
        return np.all([self.kernels[i] == b.kernels[i] for i in range(len(self.kernels))])

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return np.all([kernel.is_stationary() for kernel in self.kernels])

    @property
    def requires_vector_input(self):
        """Returns whether the kernel is defined on discrete structures."""
        return np.any([kernel.requires_vector_input for kernel in self.kernels])

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to `np.diag(self(X))`; however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X, n_kernels)
            Diagonal of kernel k(X, X)
        """
        return np.vstack([kernel.diag(X) for kernel in self.kernels]).T