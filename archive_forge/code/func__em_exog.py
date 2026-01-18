import numpy as np
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.regime_switching import markov_switching
def _em_exog(self, result, endog, exog, switching, tmp=None):
    """
        EM step for regression coefficients
        """
    k_exog = exog.shape[1]
    coeffs = np.zeros((self.k_regimes, k_exog))
    if not np.all(switching):
        nonswitching_exog = exog[:, ~switching]
        nonswitching_coeffs = np.dot(np.linalg.pinv(nonswitching_exog), endog)
        coeffs[:, ~switching] = nonswitching_coeffs
        endog = endog - np.dot(nonswitching_exog, nonswitching_coeffs)
    if np.any(switching):
        switching_exog = exog[:, switching]
        if tmp is None:
            tmp = np.sqrt(result.smoothed_marginal_probabilities)
        for i in range(self.k_regimes):
            tmp_endog = tmp[i] * endog
            tmp_exog = tmp[i][:, np.newaxis] * switching_exog
            coeffs[i, switching] = np.dot(np.linalg.pinv(tmp_exog), tmp_endog)
    return coeffs