import numpy as np
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.regime_switching import (
from statsmodels.tsa.statespace.tools import (
def _em_autoregressive(self, result, betas, tmp=None):
    """
        EM step for autoregressive coefficients and variances
        """
    if tmp is None:
        tmp = np.sqrt(result.smoothed_marginal_probabilities)
    resid = np.zeros((self.k_regimes, self.nobs + self.order))
    resid[:] = self.orig_endog
    if self._k_exog > 0:
        for i in range(self.k_regimes):
            resid[i] -= np.dot(self.orig_exog, betas[i])
    coeffs = np.zeros((self.k_regimes,) + (self.order,))
    variance = np.zeros((self.k_regimes,))
    exog = np.zeros((self.nobs, self.order))
    for i in range(self.k_regimes):
        endog = resid[i, self.order:]
        exog = lagmat(resid[i], self.order)[self.order:]
        tmp_endog = tmp[i] * endog
        tmp_exog = tmp[i][:, None] * exog
        coeffs[i] = np.dot(np.linalg.pinv(tmp_exog), tmp_endog)
        if self.switching_variance:
            tmp_resid = endog - np.dot(exog, coeffs[i])
            variance[i] = np.sum(tmp_resid ** 2 * result.smoothed_marginal_probabilities[i]) / np.sum(result.smoothed_marginal_probabilities[i])
        else:
            tmp_resid = tmp_endog - np.dot(tmp_exog, coeffs[i])
            variance[i] = np.sum(tmp_resid ** 2)
    if not self.switching_variance:
        variance = variance.sum() / self.nobs
    return (coeffs, variance)