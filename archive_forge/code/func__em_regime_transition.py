import warnings
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from statsmodels.base.data import PandasData
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, bic, hqic
from statsmodels.tools.numdiff import approx_fprime_cs, approx_hess_cs
from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tools.tools import Bunch, pinv_extended
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.regime_switching._hamilton_filter import (
from statsmodels.tsa.regime_switching._kim_smoother import (
from statsmodels.tsa.statespace.tools import (
def _em_regime_transition(self, result):
    """
        EM step for regime transition probabilities
        """
    tmp = result.smoothed_joint_probabilities
    for i in range(tmp.ndim - 3):
        tmp = np.sum(tmp, -2)
    smoothed_joint_probabilities = tmp
    k_transition = len(self.parameters[0, 'regime_transition'])
    regime_transition = np.zeros((self.k_regimes, k_transition))
    for i in range(self.k_regimes):
        for j in range(self.k_regimes - 1):
            regime_transition[i, j] = np.sum(smoothed_joint_probabilities[j, i]) / np.sum(result.smoothed_marginal_probabilities[i])
        delta = np.sum(regime_transition[i]) - 1
        if delta > 0:
            warnings.warn('Invalid regime transition probabilities estimated in EM iteration; probabilities have been re-scaled to continue estimation.', EstimationWarning)
            regime_transition[i] /= 1 + delta + 1e-06
    return regime_transition