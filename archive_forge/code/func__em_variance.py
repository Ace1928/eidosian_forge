import numpy as np
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.regime_switching import markov_switching
def _em_variance(self, result, endog, exog, betas, tmp=None):
    """
        EM step for variances
        """
    k_exog = 0 if exog is None else exog.shape[1]
    if self.switching_variance:
        variance = np.zeros(self.k_regimes)
        for i in range(self.k_regimes):
            if k_exog > 0:
                resid = endog - np.dot(exog, betas[i])
            else:
                resid = endog
            variance[i] = np.sum(resid ** 2 * result.smoothed_marginal_probabilities[i]) / np.sum(result.smoothed_marginal_probabilities[i])
    else:
        variance = 0
        if tmp is None:
            tmp = np.sqrt(result.smoothed_marginal_probabilities)
        for i in range(self.k_regimes):
            tmp_endog = tmp[i] * endog
            if k_exog > 0:
                tmp_exog = tmp[i][:, np.newaxis] * exog
                resid = tmp_endog - np.dot(tmp_exog, betas[i])
            else:
                resid = tmp_endog
            variance += np.sum(resid ** 2)
        variance /= self.nobs
    return variance