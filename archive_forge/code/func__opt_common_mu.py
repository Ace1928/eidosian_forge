import numpy as np
from .descriptive import _OptFuncts
from scipy import optimize
from scipy.stats import chi2
def _opt_common_mu(self, mu):
    """
        Optimizes the likelihood under the null hypothesis that all groups have
        mean mu.

        Parameters
        ----------
        mu : float
            The common mean.

        Returns
        -------
        llr : float
            -2 times the llr ratio, which is the test statistic.
        """
    nobs = self.nobs
    endog = self.endog
    num_groups = self.num_groups
    endog_asarray = np.zeros((nobs, num_groups))
    obs_num = 0
    for arr_num in range(len(endog)):
        new_obs_num = obs_num + len(endog[arr_num])
        endog_asarray[obs_num:new_obs_num, arr_num] = endog[arr_num] - mu
        obs_num = new_obs_num
    est_vect = endog_asarray
    wts = np.ones(est_vect.shape[0]) * (1.0 / est_vect.shape[0])
    eta_star = self._modif_newton(np.zeros(num_groups), est_vect, wts)
    denom = 1.0 + np.dot(eta_star, est_vect.T)
    self.new_weights = 1.0 / nobs * 1.0 / denom
    llr = np.sum(np.log(nobs * self.new_weights))
    return -2 * llr