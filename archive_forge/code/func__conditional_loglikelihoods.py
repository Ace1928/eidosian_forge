import numpy as np
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.regime_switching import (
from statsmodels.tsa.statespace.tools import (
def _conditional_loglikelihoods(self, params):
    """
        Compute loglikelihoods conditional on the current period's regime and
        the last `self.order` regimes.
        """
    resid = self._resid(params)
    variance = params[self.parameters['variance']].squeeze()
    if self.switching_variance:
        variance = np.reshape(variance, (self.k_regimes, 1, 1))
    conditional_loglikelihoods = -0.5 * resid ** 2 / variance - 0.5 * np.log(2 * np.pi * variance)
    return conditional_loglikelihoods