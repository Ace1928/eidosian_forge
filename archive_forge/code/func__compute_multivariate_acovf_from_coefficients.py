import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def _compute_multivariate_acovf_from_coefficients(coefficients, error_variance, maxlag=None, forward_autocovariances=False):
    """
    Compute multivariate autocovariances from vector autoregression coefficient
    matrices

    Parameters
    ----------
    coefficients : array or list
        The coefficients matrices. If a list, should be a list of length
        `order`, where each element is an array sized `k_endog` x `k_endog`. If
        an array, should be the coefficient matrices horizontally concatenated
        and sized `k_endog` x `k_endog * order`.
    error_variance : ndarray
        The variance / covariance matrix of the error term. Should be sized
        `k_endog` x `k_endog`.
    maxlag : int, optional
        The maximum autocovariance to compute. Default is `order`-1. Can be
        zero, in which case it returns the variance.
    forward_autocovariances : bool, optional
        Whether or not to compute forward autocovariances
        :math:`E(y_t y_{t+j}')`. Default is False, so that backward
        autocovariances :math:`E(y_t y_{t-j}')` are returned.

    Returns
    -------
    autocovariances : list
        A list of the first `maxlag` autocovariance matrices. Each matrix is
        shaped `k_endog` x `k_endog`.

    Notes
    -----
    Computes

    .. math::

        \\Gamma(j) = E(y_t y_{t-j}')

    for j = 1, ..., `maxlag`, unless `forward_autocovariances` is specified,
    in which case it computes:

    .. math::

        E(y_t y_{t+j}') = \\Gamma(j)'

    Coefficients are assumed to be provided from the VAR model:

    .. math::
        y_t = A_1 y_{t-1} + \\dots + A_p y_{t-p} + \\varepsilon_t

    Autocovariances are calculated by solving the associated discrete Lyapunov
    equation of the state space representation of the VAR process.
    """
    from scipy import linalg
    if type(coefficients) is list:
        order = len(coefficients)
        k_endog = coefficients[0].shape[0]
    else:
        k_endog, order = coefficients.shape
        order //= k_endog
        coefficients = [coefficients[:k_endog, i * k_endog:(i + 1) * k_endog] for i in range(order)]
    if maxlag is None:
        maxlag = order - 1
    companion = companion_matrix([1] + [-np.squeeze(coefficients[i]) for i in range(order)]).T
    selected_variance = np.zeros(companion.shape)
    selected_variance[:k_endog, :k_endog] = error_variance
    stacked_cov = linalg.solve_discrete_lyapunov(companion, selected_variance)
    autocovariances = [stacked_cov[:k_endog, i * k_endog:(i + 1) * k_endog] for i in range(min(order, maxlag + 1))]
    for i in range(maxlag - (order - 1)):
        stacked_cov = np.dot(companion, stacked_cov)
        autocovariances += [stacked_cov[:k_endog, -k_endog:]]
    if forward_autocovariances:
        for i in range(len(autocovariances)):
            autocovariances[i] = autocovariances[i].T
    return autocovariances