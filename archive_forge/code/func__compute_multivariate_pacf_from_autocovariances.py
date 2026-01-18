import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def _compute_multivariate_pacf_from_autocovariances(autocovariances, order=None, k_endog=None):
    """
    Compute multivariate partial autocorrelations from autocovariances.

    Parameters
    ----------
    autocovariances : list
        Autocorrelations matrices. Should be a list of length `order` + 1,
        where each element is an array sized `k_endog` x `k_endog`.
    order : int, optional
        The order of the autoregression.
    k_endog : int, optional
        The dimension of the data vector.

    Returns
    -------
    pacf : list
        List of first `order` multivariate partial autocorrelations.

    See Also
    --------
    unconstrain_stationary_multivariate

    Notes
    -----
    Note that this computes multivariate partial autocorrelations.

    Corresponds to the inverse of Lemma 2.1 in Ansley and Kohn (1986). See
    `unconstrain_stationary_multivariate` for more details.

    Computes sample partial autocorrelations if sample autocovariances are
    given.
    """
    from scipy import linalg
    if order is None:
        order = len(autocovariances) - 1
    if k_endog is None:
        k_endog = autocovariances[0].shape[0]
    forward_variances = []
    backward_variances = []
    forwards = []
    backwards = []
    forward_factors = []
    backward_factors = []
    partial_autocorrelations = []
    for s in range(order):
        prev_forwards = list(forwards)
        prev_backwards = list(backwards)
        forwards = []
        backwards = []
        forward_variance = autocovariances[0].copy()
        backward_variance = autocovariances[0].T.copy()
        for k in range(s):
            forward_variance -= np.dot(prev_forwards[k], autocovariances[k + 1])
            backward_variance -= np.dot(prev_backwards[k], autocovariances[k + 1].T)
        forward_variances.append(forward_variance)
        backward_variances.append(backward_variance)
        forward_factors.append(linalg.cholesky(forward_variances[s], lower=True))
        backward_factors.append(linalg.cholesky(backward_variances[s], lower=True))
        if s == 0:
            forwards.append(linalg.cho_solve((forward_factors[0], True), autocovariances[1]).T)
            backwards.append(linalg.cho_solve((backward_factors[0], True), autocovariances[1].T).T)
        else:
            tmp_sum = autocovariances[s + 1].T.copy()
            for k in range(s):
                tmp_sum -= np.dot(prev_forwards[k], autocovariances[s - k].T)
            forwards.append(linalg.cho_solve((backward_factors[s], True), tmp_sum.T).T)
            backwards.append(linalg.cho_solve((forward_factors[s], True), tmp_sum).T)
        for k in range(s):
            forwards.insert(k, prev_forwards[k] - np.dot(forwards[-1], prev_backwards[s - (k + 1)]))
            backwards.insert(k, prev_backwards[k] - np.dot(backwards[-1], prev_forwards[s - (k + 1)]))
        partial_autocorrelations.append(linalg.solve_triangular(forward_factors[s], np.dot(forwards[s], backward_factors[s]), lower=True))
    return partial_autocorrelations