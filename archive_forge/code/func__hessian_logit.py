import warnings
import numpy as np
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.discrete.discrete_model import (DiscreteModel, CountModel,
from statsmodels.distributions import zipoisson, zigenpoisson, zinegbin
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.compat.pandas import Appender
def _hessian_logit(self, params):
    params_infl = params[:self.k_inflate]
    params_main = params[self.k_inflate:]
    y = self.endog
    w = self.model_infl.predict(params_infl)
    w = np.clip(w, np.finfo(float).eps, 1 - np.finfo(float).eps)
    score_main = self.model_main.score_obs(params_main)
    llf_main = self.model_main.loglikeobs(params_main)
    llf = self.loglikeobs(params)
    zero_idx = np.nonzero(y == 0)[0]
    nonzero_idx = np.nonzero(y)[0]
    hess_arr = np.zeros((self.k_inflate, self.k_exog + self.k_inflate))
    pmf = np.exp(llf)
    for i in range(self.k_inflate):
        for j in range(i, -1, -1):
            hess_arr[i, j] = (self.exog_infl[zero_idx, i] * self.exog_infl[zero_idx, j] * (w[zero_idx] * (1 - w[zero_idx]) * ((1 - np.exp(llf_main[zero_idx])) * (1 - 2 * w[zero_idx]) * np.exp(llf[zero_idx]) - (w[zero_idx] - w[zero_idx] ** 2) * (1 - np.exp(llf_main[zero_idx])) ** 2) / pmf[zero_idx] ** 2)).sum() - (self.exog_infl[nonzero_idx, i] * self.exog_infl[nonzero_idx, j] * w[nonzero_idx] * (1 - w[nonzero_idx])).sum()
    for i in range(self.k_inflate):
        for j in range(self.k_exog):
            hess_arr[i, j + self.k_inflate] = -(score_main[zero_idx, j] * w[zero_idx] * (1 - w[zero_idx]) * self.exog_infl[zero_idx, i] / pmf[zero_idx]).sum()
    return hess_arr