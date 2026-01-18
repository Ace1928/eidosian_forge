import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy.special import logsumexp
from .base import BaseEstimator, ClassifierMixin, _fit_context
from .preprocessing import LabelBinarizer, binarize, label_binarize
from .utils._param_validation import Interval
from .utils.extmath import safe_sparse_dot
from .utils.multiclass import _check_partial_fit_first_call
from .utils.validation import _check_sample_weight, check_is_fitted, check_non_negative
@staticmethod
def _update_mean_variance(n_past, mu, var, X, sample_weight=None):
    """Compute online update of Gaussian mean and variance.

        Given starting sample count, mean, and variance, a new set of
        points X, and optionally sample weights, return the updated mean and
        variance. (NB - each dimension (column) in X is treated as independent
        -- you get variance, not covariance).

        Can take scalar mean and variance, or vector mean and variance to
        simultaneously update a number of independent Gaussians.

        See Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:

        http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

        Parameters
        ----------
        n_past : int
            Number of samples represented in old mean and variance. If sample
            weights were given, this should contain the sum of sample
            weights represented in old mean and variance.

        mu : array-like of shape (number of Gaussians,)
            Means for Gaussians in original set.

        var : array-like of shape (number of Gaussians,)
            Variances for Gaussians in original set.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        total_mu : array-like of shape (number of Gaussians,)
            Updated mean for each Gaussian over the combined set.

        total_var : array-like of shape (number of Gaussians,)
            Updated variance for each Gaussian over the combined set.
        """
    if X.shape[0] == 0:
        return (mu, var)
    if sample_weight is not None:
        n_new = float(sample_weight.sum())
        if np.isclose(n_new, 0.0):
            return (mu, var)
        new_mu = np.average(X, axis=0, weights=sample_weight)
        new_var = np.average((X - new_mu) ** 2, axis=0, weights=sample_weight)
    else:
        n_new = X.shape[0]
        new_var = np.var(X, axis=0)
        new_mu = np.mean(X, axis=0)
    if n_past == 0:
        return (new_mu, new_var)
    n_total = float(n_past + n_new)
    total_mu = (n_new * new_mu + n_past * mu) / n_total
    old_ssd = n_past * var
    new_ssd = n_new * new_var
    total_ssd = old_ssd + new_ssd + n_new * n_past / n_total * (mu - new_mu) ** 2
    total_var = total_ssd / n_total
    return (total_mu, total_var)