import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.stats.distributions import norm
from statsmodels.base._penalties import Penalty
import statsmodels.base.model as base
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
def _make_param_names(self, exog_re):
    """
        Returns the full parameter names list, just the exogenous random
        effects variables, and the exogenous random effects variables with
        the interaction terms.
        """
    exog_names = list(self.exog_names)
    exog_re_names = _get_exog_re_names(self, exog_re)
    param_names = []
    jj = self.k_fe
    for i in range(len(exog_re_names)):
        for j in range(i + 1):
            if i == j:
                param_names.append(exog_re_names[i] + ' Var')
            else:
                param_names.append(exog_re_names[j] + ' x ' + exog_re_names[i] + ' Cov')
            jj += 1
    vc_names = [x + ' Var' for x in self.exog_vc.names]
    return (exog_names + param_names + vc_names, exog_re_names, param_names)