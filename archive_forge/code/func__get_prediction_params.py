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
def _get_prediction_params(self, start_idx):
    """
        Returns internal parameter representation of smoothing parameters and
        "initial" states for prediction/simulation, that is the states just
        before the first prediction/simulation step.
        """
    internal_params = self.model._internal_params(self.params)
    if start_idx == 0:
        return internal_params
    else:
        internal_states = self.model._get_internal_states(self.states, self.params)
        start_state = np.empty(6 + self.seasonal_periods)
        start_state[0:4] = internal_params[0:4]
        start_state[4:] = internal_states[start_idx - 1, :]
        return start_state