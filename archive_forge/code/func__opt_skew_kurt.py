import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
def _opt_skew_kurt(self, nuis_params):
    """
        Called by test_joint_skew_kurt.  This function is optimized over
        nuisance parameters mu and sigma

        Parameters
        ----------
        nuis_params : 1darray
            An array with a nuisance mean and variance parameter

        Returns
        ------
        llr : float
            The log likelihood ratio of a pre-speified skewness and
            kurtosis holding the nuisance parameters constant.
        """
    endog = self.endog
    nobs = self.nobs
    mu_data = endog - nuis_params[0]
    sig_data = (endog - nuis_params[0]) ** 2 - nuis_params[1]
    skew_data = (endog - nuis_params[0]) ** 3 / nuis_params[1] ** 1.5 - self.skew0
    kurt_data = (endog - nuis_params[0]) ** 4 / nuis_params[1] ** 2 - 3 - self.kurt0
    est_vect = np.column_stack((mu_data, sig_data, skew_data, kurt_data))
    eta_star = self._modif_newton(np.array([1.0 / nobs, 1.0 / nobs, 1.0 / nobs, 1.0 / nobs]), est_vect, np.ones(nobs) * (1.0 / nobs))
    denom = 1.0 + np.dot(eta_star, est_vect.T)
    self.new_weights = 1.0 / nobs * 1.0 / denom
    llr = np.sum(np.log(nobs * self.new_weights))
    return -2 * llr