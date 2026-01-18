import warnings
from functools import partial
from numbers import Integral
import numpy as np
from scipy import linalg, sparse
from ..utils import deprecated
from ..utils._param_validation import Interval, StrOptions, validate_params
from . import check_random_state
from ._array_api import _is_numpy_namespace, device, get_namespace
from .sparsefuncs_fast import csr_row_norms
from .validation import check_array
def _incremental_mean_and_var(X, last_mean, last_variance, last_sample_count, sample_weight=None):
    """Calculate mean update and a Youngs and Cramer variance update.

    If sample_weight is given, the weighted mean and variance is computed.

    Update a given mean and (possibly) variance according to new data given
    in X. last_mean is always required to compute the new mean.
    If last_variance is None, no variance is computed and None return for
    updated_variance.

    From the paper "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to use for variance update.

    last_mean : array-like of shape (n_features,)

    last_variance : array-like of shape (n_features,)

    last_sample_count : array-like of shape (n_features,)
        The number of samples encountered until now if sample_weight is None.
        If sample_weight is not None, this is the sum of sample_weight
        encountered.

    sample_weight : array-like of shape (n_samples,) or None
        Sample weights. If None, compute the unweighted mean/variance.

    Returns
    -------
    updated_mean : ndarray of shape (n_features,)

    updated_variance : ndarray of shape (n_features,)
        None if last_variance was None.

    updated_sample_count : ndarray of shape (n_features,)

    Notes
    -----
    NaNs are ignored during the algorithm.

    References
    ----------
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
        variance: recommendations, The American Statistician, Vol. 37, No. 3,
        pp. 242-247

    Also, see the sparse implementation of this in
    `utils.sparsefuncs.incr_mean_variance_axis` and
    `utils.sparsefuncs_fast.incr_mean_variance_axis0`
    """
    last_sum = last_mean * last_sample_count
    X_nan_mask = np.isnan(X)
    if np.any(X_nan_mask):
        sum_op = np.nansum
    else:
        sum_op = np.sum
    if sample_weight is not None:
        new_sum = _safe_accumulator_op(np.matmul, sample_weight, np.where(X_nan_mask, 0, X))
        new_sample_count = _safe_accumulator_op(np.sum, sample_weight[:, None] * ~X_nan_mask, axis=0)
    else:
        new_sum = _safe_accumulator_op(sum_op, X, axis=0)
        n_samples = X.shape[0]
        new_sample_count = n_samples - np.sum(X_nan_mask, axis=0)
    updated_sample_count = last_sample_count + new_sample_count
    updated_mean = (last_sum + new_sum) / updated_sample_count
    if last_variance is None:
        updated_variance = None
    else:
        T = new_sum / new_sample_count
        temp = X - T
        if sample_weight is not None:
            correction = _safe_accumulator_op(np.matmul, sample_weight, np.where(X_nan_mask, 0, temp))
            temp **= 2
            new_unnormalized_variance = _safe_accumulator_op(np.matmul, sample_weight, np.where(X_nan_mask, 0, temp))
        else:
            correction = _safe_accumulator_op(sum_op, temp, axis=0)
            temp **= 2
            new_unnormalized_variance = _safe_accumulator_op(sum_op, temp, axis=0)
        new_unnormalized_variance -= correction ** 2 / new_sample_count
        last_unnormalized_variance = last_variance * last_sample_count
        with np.errstate(divide='ignore', invalid='ignore'):
            last_over_new_count = last_sample_count / new_sample_count
            updated_unnormalized_variance = last_unnormalized_variance + new_unnormalized_variance + last_over_new_count / updated_sample_count * (last_sum / last_over_new_count - new_sum) ** 2
        zeros = last_sample_count == 0
        updated_unnormalized_variance[zeros] = new_unnormalized_variance[zeros]
        updated_variance = updated_unnormalized_variance / updated_sample_count
    return (updated_mean, updated_variance, updated_sample_count)