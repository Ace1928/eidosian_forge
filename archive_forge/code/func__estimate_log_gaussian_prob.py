import numpy as np
from scipy import linalg
from ..utils import check_array
from ..utils._param_validation import StrOptions
from ..utils.extmath import row_norms
from ._base import BaseMixture, _check_shape
def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type):
    """Estimate the log Gaussian probability.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    means : array-like of shape (n_components, n_features)

    precisions_chol : array-like
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    log_det = _compute_log_det_cholesky(precisions_chol, covariance_type, n_features)
    if covariance_type == 'full':
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)
    elif covariance_type == 'tied':
        log_prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)
    elif covariance_type == 'diag':
        precisions = precisions_chol ** 2
        log_prob = np.sum(means ** 2 * precisions, 1) - 2.0 * np.dot(X, (means * precisions).T) + np.dot(X ** 2, precisions.T)
    elif covariance_type == 'spherical':
        precisions = precisions_chol ** 2
        log_prob = np.sum(means ** 2, 1) * precisions - 2 * np.dot(X, means.T * precisions) + np.outer(row_norms(X, squared=True), precisions)
    return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det