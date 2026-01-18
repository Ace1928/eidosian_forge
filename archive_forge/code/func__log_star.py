import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
def _log_star(self, eta, est_vect, weights, nobs):
    """
        Transforms the log of observation probabilities in terms of the
        Lagrange multiplier to the log 'star' of the probabilities.

        Parameters
        ----------
        eta : float
            Lagrange multiplier

        est_vect : ndarray (n,k)
            Estimating equations vector

        wts : nx1 array
            Observation weights

        Returns
        ------
        data_star : ndarray
            The weighted logstar of the estimting equations

        Notes
        -----
        This function is only a placeholder for the _fit_Newton.
        The function value is not used in optimization and the optimal value
        is disregarded when computing the log likelihood ratio.
        """
    data_star = np.log(weights) + (np.sum(weights) + np.dot(est_vect, eta))
    idx = data_star < 1.0 / nobs
    not_idx = ~idx
    nx = nobs * data_star[idx]
    data_star[idx] = np.log(1.0 / nobs) - 1.5 + nx * (2.0 - nx / 2)
    data_star[not_idx] = np.log(data_star[not_idx])
    return data_star