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
def _setup_bounds(self):
    lb = np.zeros(self._k_params_internal) + 0.0001
    ub = np.ones(self._k_params_internal) - 0.0001
    lb[3], ub[3] = (0.8, 0.98)
    if self.initialization_method == 'estimated':
        lb[4:-1] = -np.inf
        ub[4:-1] = np.inf
        if self.seasonal == 'mul':
            lb[-1], ub[-1] = (1, 1)
        else:
            lb[-1], ub[-1] = (0, 0)
    for p in self._internal_param_names:
        idx = self._internal_params_index[p]
        if p in self.bounds:
            lb[idx], ub[idx] = self.bounds[p]
    return [(lb[i], ub[i]) for i in range(self._k_params_internal)]