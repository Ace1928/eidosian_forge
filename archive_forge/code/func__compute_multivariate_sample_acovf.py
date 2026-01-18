import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def _compute_multivariate_sample_acovf(endog, maxlag):
    """
    Computer multivariate sample autocovariances

    Parameters
    ----------
    endog : array_like
        Sample data on which to compute sample autocovariances. Shaped
        `nobs` x `k_endog`.
    maxlag : int
        Maximum lag to use when computing the sample autocovariances.

    Returns
    -------
    sample_autocovariances : list
        A list of the first `maxlag` sample autocovariance matrices. Each
        matrix is shaped `k_endog` x `k_endog`.

    Notes
    -----
    This function computes the forward sample autocovariances:

    .. math::

        \\hat \\Gamma(s) = \\frac{1}{n} \\sum_{t=1}^{n-s}
        (Z_t - \\bar Z) (Z_{t+s} - \\bar Z)'

    See page 353 of Wei (1990). This function is primarily implemented for
    checking the partial autocorrelation functions below, and so is quite slow.

    References
    ----------
    .. [*] Wei, William. 1990.
       Time Series Analysis : Univariate and Multivariate Methods. Boston:
       Pearson.
    """
    endog = np.array(endog)
    if endog.ndim == 1:
        endog = endog[:, np.newaxis]
    endog -= np.mean(endog, axis=0)
    nobs, k_endog = endog.shape
    sample_autocovariances = []
    for s in range(maxlag + 1):
        sample_autocovariances.append(np.zeros((k_endog, k_endog)))
        for t in range(nobs - s):
            sample_autocovariances[s] += np.outer(endog[t], endog[t + s])
        sample_autocovariances[s] /= nobs
    return sample_autocovariances