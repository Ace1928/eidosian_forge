import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
def _hess(self, eta, est_vect, weights, nobs):
    """
        Calculates the hessian of a weighted empirical likelihood
        problem.

        Parameters
        ----------
        eta : ndarray, (1,m)
            Lagrange multiplier in the profile likelihood maximization

        est_vect : ndarray (n,k)
            Estimating equations vector

        weights : 1darray
            Observation weights

        Returns
        -------
        hess : m x m array
            Weighted hessian used in _wtd_modif_newton
        """
    data_star_doub_prime = np.sum(weights) + np.dot(est_vect, eta)
    idx = data_star_doub_prime < 1.0 / nobs
    not_idx = ~idx
    data_star_doub_prime[idx] = -nobs ** 2
    data_star_doub_prime[not_idx] = -data_star_doub_prime[not_idx] ** (-2)
    wtd_dsdp = weights * data_star_doub_prime
    return np.dot(est_vect.T, wtd_dsdp[:, None] * est_vect)