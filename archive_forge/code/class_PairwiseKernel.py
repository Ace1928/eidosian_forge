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
class PairwiseKernel(Kernel):
    """Wrapper for kernels in sklearn.metrics.pairwise.

    A thin wrapper around the functionality of the kernels in
    sklearn.metrics.pairwise.

    Note: Evaluation of eval_gradient is not analytic but numeric and all
          kernels support only isotropic distances. The parameter gamma is
          considered to be a hyperparameter and may be optimized. The other
          kernel parameters are set directly at initialization and are kept
          fixed.

    .. versionadded:: 0.18

    Parameters
    ----------
    gamma : float, default=1.0
        Parameter gamma of the pairwise kernel specified by metric. It should
        be positive.

    gamma_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'gamma'.
        If set to "fixed", 'gamma' cannot be changed during
        hyperparameter tuning.

    metric : {"linear", "additive_chi2", "chi2", "poly", "polynomial",               "rbf", "laplacian", "sigmoid", "cosine"} or callable,               default="linear"
        The metric to use when calculating kernel between instances in a
        feature array. If metric is a string, it must be one of the metrics
        in pairwise.PAIRWISE_KERNEL_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a kernel matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.

    pairwise_kernels_kwargs : dict, default=None
        All entries of this dict (if any) are passed as keyword arguments to
        the pairwise kernel function.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import PairwiseKernel
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = PairwiseKernel(metric='rbf')
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9733...
    >>> gpc.predict_proba(X[:2,:])
    array([[0.8880..., 0.05663..., 0.05532...],
           [0.8676..., 0.07073..., 0.06165...]])
    """

    def __init__(self, gamma=1.0, gamma_bounds=(1e-05, 100000.0), metric='linear', pairwise_kernels_kwargs=None):
        self.gamma = gamma
        self.gamma_bounds = gamma_bounds
        self.metric = metric
        self.pairwise_kernels_kwargs = pairwise_kernels_kwargs

    @property
    def hyperparameter_gamma(self):
        return Hyperparameter('gamma', 'numeric', self.gamma_bounds)

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

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        pairwise_kernels_kwargs = self.pairwise_kernels_kwargs
        if self.pairwise_kernels_kwargs is None:
            pairwise_kernels_kwargs = {}
        X = np.atleast_2d(X)
        K = pairwise_kernels(X, Y, metric=self.metric, gamma=self.gamma, filter_params=True, **pairwise_kernels_kwargs)
        if eval_gradient:
            if self.hyperparameter_gamma.fixed:
                return (K, np.empty((X.shape[0], X.shape[0], 0)))
            else:

                def f(gamma):
                    return pairwise_kernels(X, Y, metric=self.metric, gamma=np.exp(gamma), filter_params=True, **pairwise_kernels_kwargs)
                return (K, _approx_fprime(self.theta, f, 1e-10))
        else:
            return K

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return np.apply_along_axis(self, 1, X).ravel()

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return self.metric in ['rbf']

    def __repr__(self):
        return '{0}(gamma={1}, metric={2})'.format(self.__class__.__name__, self.gamma, self.metric)