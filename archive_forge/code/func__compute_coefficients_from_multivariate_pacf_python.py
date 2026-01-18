import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def _compute_coefficients_from_multivariate_pacf_python(partial_autocorrelations, error_variance, transform_variance=False, order=None, k_endog=None):
    """
    Transform matrices with singular values less than one to matrices
    corresponding to a stationary (or invertible) process.

    Parameters
    ----------
    partial_autocorrelations : list
        Partial autocorrelation matrices. Should be a list of length `order`,
        where each element is an array sized `k_endog` x `k_endog`.
    error_variance : ndarray
        The variance / covariance matrix of the error term. Should be sized
        `k_endog` x `k_endog`. This is used as input in the algorithm even if
        is not transformed by it (when `transform_variance` is False). The
        error term variance is required input when transformation is used
        either to force an autoregressive component to be stationary or to
        force a moving average component to be invertible.
    transform_variance : bool, optional
        Whether or not to transform the error variance term. This option is
        not typically used, and the default is False.
    order : int, optional
        The order of the autoregression.
    k_endog : int, optional
        The dimension of the data vector.

    Returns
    -------
    coefficient_matrices : list
        Transformed coefficient matrices leading to a stationary VAR
        representation.

    See Also
    --------
    constrain_stationary_multivariate

    Notes
    -----
    Corresponds to Lemma 2.1 in Ansley and Kohn (1986). See
    `constrain_stationary_multivariate` for more details.
    """
    from scipy import linalg
    if order is None:
        order = len(partial_autocorrelations)
    if k_endog is None:
        k_endog = partial_autocorrelations[0].shape[0]
    if not transform_variance:
        initial_variance = error_variance
        error_variance = np.eye(k_endog) * (order + k_endog) ** 10
    forward_variances = [error_variance]
    backward_variances = [error_variance]
    autocovariances = [error_variance]
    forwards = []
    backwards = []
    error_variance_factor = linalg.cholesky(error_variance, lower=True)
    forward_factors = [error_variance_factor]
    backward_factors = [error_variance_factor]
    for s in range(order):
        prev_forwards = forwards
        prev_backwards = backwards
        forwards = []
        backwards = []
        forwards.append(linalg.solve_triangular(backward_factors[s], partial_autocorrelations[s].T, lower=True, trans='T'))
        forwards[0] = np.dot(forward_factors[s], forwards[0].T)
        backwards.append(linalg.solve_triangular(forward_factors[s], partial_autocorrelations[s], lower=True, trans='T'))
        backwards[0] = np.dot(backward_factors[s], backwards[0].T)
        tmp = np.dot(forwards[0], backward_variances[s])
        autocovariances.append(tmp.copy().T)
        for k in range(s):
            forwards.insert(k, prev_forwards[k] - np.dot(forwards[-1], prev_backwards[s - (k + 1)]))
            backwards.insert(k, prev_backwards[k] - np.dot(backwards[-1], prev_forwards[s - (k + 1)]))
            autocovariances[s + 1] += np.dot(autocovariances[k + 1], prev_forwards[s - (k + 1)].T)
        forward_variances.append(forward_variances[s] - np.dot(tmp, forwards[s].T))
        backward_variances.append(backward_variances[s] - np.dot(np.dot(backwards[s], forward_variances[s]), backwards[s].T))
        forward_factors.append(linalg.cholesky(forward_variances[s + 1], lower=True))
        backward_factors.append(linalg.cholesky(backward_variances[s + 1], lower=True))
    variance = forward_variances[-1]
    if not transform_variance:
        initial_variance_factor = np.linalg.cholesky(initial_variance)
        transformed_variance_factor = np.linalg.cholesky(variance)
        transform = np.dot(initial_variance_factor, np.linalg.inv(transformed_variance_factor))
        inv_transform = np.linalg.inv(transform)
        for i in range(order):
            forwards[i] = np.dot(np.dot(transform, forwards[i]), inv_transform)
    return (forwards, variance)