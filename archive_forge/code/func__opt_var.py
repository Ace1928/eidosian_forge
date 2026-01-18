import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
def _opt_var(self, nuisance_mu, pval=False):
    """
        This is the function to be optimized over a nuisance mean parameter
        to determine the likelihood ratio for the variance

        Parameters
        ----------
        nuisance_mu : float
            Value of a nuisance mean parameter

        Returns
        -------
        llr : float
            Log likelihood of a pre-specified variance holding the nuisance
            parameter constant
        """
    endog = self.endog
    nobs = self.nobs
    sig_data = (endog - nuisance_mu) ** 2 - self.sig2_0
    mu_data = endog - nuisance_mu
    est_vect = np.column_stack((mu_data, sig_data))
    eta_star = self._modif_newton(np.array([1.0 / nobs, 1.0 / nobs]), est_vect, np.ones(nobs) * (1.0 / nobs))
    denom = 1 + np.dot(eta_star, est_vect.T)
    self.new_weights = 1.0 / nobs * 1.0 / denom
    llr = np.sum(np.log(nobs * self.new_weights))
    if pval:
        return chi2.sf(-2 * llr, 1)
    return -2 * llr