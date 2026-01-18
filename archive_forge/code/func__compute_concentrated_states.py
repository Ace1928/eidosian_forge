import numpy as np
import pandas as pd
from statsmodels.base.data import PandasData
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.validation import (array_like, bool_like, float_like,
from statsmodels.tsa.exponential_smoothing import initialization as es_init
from statsmodels.tsa.statespace import initialization as ss_init
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.compat.pandas import Appender
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import forg
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
def _compute_concentrated_states(self, params, *args, **kwargs):
    kwargs['conserve_memory'] = MEMORY_CONSERVE & ~MEMORY_NO_FORECAST
    super().loglike(params, *args, **kwargs)
    y_tilde = np.array(self.ssm._kalman_filter.forecast_error[0], copy=True)
    T = self['transition', 1:, 1:]
    R = self['selection', 1:]
    Z = self['design', :, 1:].copy()
    i = 1
    if self.trend:
        Z[0, i] = 1.0
        i += 1
    if self.seasonal:
        Z[0, i] = 0.0
        Z[0, -1] = 1.0
    D = T - R.dot(Z)
    w = np.zeros((self.nobs, self.k_states - 1), dtype=D.dtype)
    w[0] = Z
    for i in range(self.nobs - 1):
        w[i + 1] = w[i].dot(D)
    mod_ols = GLM(y_tilde, w)
    if self.seasonal:
        R = np.zeros_like(Z)
        R[0, -self.seasonal_periods:] = 1.0
        q = np.zeros((1, 1))
        res_ols = mod_ols.fit_constrained((R, q))
    else:
        res_ols = mod_ols.fit()
    initial_level = res_ols.params[0]
    initial_trend = res_ols.params[1] if self.trend else None
    initial_seasonal = res_ols.params[-self.seasonal_periods:] if self.seasonal else None
    return (initial_level, initial_trend, initial_seasonal)