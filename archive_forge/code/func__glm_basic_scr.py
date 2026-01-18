import pandas as pd
import patsy
import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.compat.pandas import Appender
def _glm_basic_scr(result, exog, alpha):
    """
    The basic SCR from (Sun et al. Annals of Statistics 2000).

    Computes simultaneous confidence regions (SCR).

    Parameters
    ----------
    result : results instance
        The fitted GLM results instance
    exog : array_like
        The exog values spanning the interval
    alpha : float
        `1 - alpha` is the coverage probability.

    Returns
    -------
    An array with two columns, containing the lower and upper
    confidence bounds, respectively.

    Notes
    -----
    The rows of `exog` should be a sequence of covariate values
    obtained by taking one 'free variable' x and varying it over an
    interval.  The matrix `exog` is thus the basis functions and any
    other covariates evaluated as x varies.
    """
    model = result.model
    n = model.exog.shape[0]
    cov = result.cov_params()
    hess = np.linalg.inv(cov)
    A = hess / n
    B = np.linalg.cholesky(A).T
    sigma2 = (np.dot(exog, cov) * exog).sum(1)
    sigma = np.asarray(np.sqrt(sigma2))
    bz = np.linalg.solve(B.T, exog.T).T
    bz /= np.sqrt(n)
    bz /= sigma[:, None]
    bzd = np.diff(bz, 1, axis=0)
    bzdn = (bzd ** 2).sum(1)
    kappa_0 = np.sqrt(bzdn).sum()
    from scipy.stats.distributions import norm

    def func(c):
        return kappa_0 * np.exp(-c ** 2 / 2) / np.pi + 2 * (1 - norm.cdf(c)) - alpha
    from scipy.optimize import brentq
    c, rslt = brentq(func, 1, 10, full_output=True)
    if not rslt.converged:
        raise ValueError('Root finding error in basic SCR')
    return (sigma, c)