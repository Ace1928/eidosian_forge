from collections import OrderedDict
import contextlib
import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import norm, rv_continuous, rv_discrete
from scipy.stats.distributions import rv_frozen
from statsmodels.base.covtype import descriptions
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import forg
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import (
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.exponential_smoothing import base
import statsmodels.tsa.exponential_smoothing._ets_smooth as smooth
from statsmodels.tsa.exponential_smoothing.initialization import (
from statsmodels.tsa.tsatools import freq_to_period
def _get_internal_states(self, states, params):
    """
        Converts a state matrix/dataframe to the (nobs, 2+m) matrix used
        internally
        """
    internal_params = self._internal_params(params)
    if isinstance(states, (pd.Series, pd.DataFrame)):
        states = states.values
    internal_states = np.zeros((self.nobs, 2 + self.seasonal_periods))
    internal_states[:, 0] = states[:, 0]
    if self.has_trend:
        internal_states[:, 1] = states[:, 1]
    if self.has_seasonal:
        for j in range(self.seasonal_periods):
            internal_states[j:, 2 + j] = states[0:self.nobs - j, self._seasonal_index]
            internal_states[0:j, 2 + j] = internal_params[6:6 + j][::-1]
    return internal_states