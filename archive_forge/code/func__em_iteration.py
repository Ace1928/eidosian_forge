import numpy as np
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.regime_switching import (
from statsmodels.tsa.statespace.tools import (
def _em_iteration(self, params0):
    """
        EM iteration
        """
    result, params1 = markov_switching.MarkovSwitching._em_iteration(self, params0)
    tmp = np.sqrt(result.smoothed_marginal_probabilities)
    coeffs = None
    if self._k_exog > 0:
        coeffs = self._em_exog(result, self.endog, self.exog, self.parameters.switching['exog'], tmp)
        for i in range(self.k_regimes):
            params1[self.parameters[i, 'exog']] = coeffs[i]
    if self.order > 0:
        if self._k_exog > 0:
            ar_coeffs, variance = self._em_autoregressive(result, coeffs)
        else:
            ar_coeffs = self._em_exog(result, self.endog, self.exog_ar, self.parameters.switching['autoregressive'])
            variance = self._em_variance(result, self.endog, self.exog_ar, ar_coeffs, tmp)
        for i in range(self.k_regimes):
            params1[self.parameters[i, 'autoregressive']] = ar_coeffs[i]
        params1[self.parameters['variance']] = variance
    return (result, params1)