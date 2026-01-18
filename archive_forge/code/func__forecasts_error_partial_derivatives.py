from statsmodels.compat.pandas import is_int_index
import contextlib
import warnings
import datetime as dt
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tools.tools import pinv_extended, Bunch
from statsmodels.tools.sm_exceptions import PrecisionWarning, ValueWarning
from statsmodels.tools.numdiff import (_get_epsilon, approx_hess_cs,
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, aicc, bic, hqic
import statsmodels.base.wrapper as wrap
import statsmodels.tsa.base.prediction as pred
from statsmodels.base.data import PandasData
import statsmodels.tsa.base.tsa_model as tsbase
from .news import NewsResults
from .simulation_smoother import SimulationSmoother
from .kalman_smoother import SmootherResults
from .kalman_filter import INVERT_UNIVARIATE, SOLVE_LU, MEMORY_CONSERVE
from .initialization import Initialization
from .tools import prepare_exog, concat, _safe_cond, get_impact_dates
def _forecasts_error_partial_derivatives(self, params, transformed=True, includes_fixed=False, approx_complex_step=None, approx_centered=False, res=None, **kwargs):
    params = np.array(params, ndmin=1)
    if approx_complex_step is None:
        approx_complex_step = transformed
    if not transformed and approx_complex_step:
        raise ValueError('Cannot use complex-step approximations to calculate the observed_information_matrix with untransformed parameters.')
    if approx_complex_step:
        kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU
    if res is None:
        self.update(params, transformed=transformed, includes_fixed=includes_fixed, complex_step=approx_complex_step)
        res = self.ssm.filter(complex_step=approx_complex_step, **kwargs)
    n = len(params)
    partials_forecasts_error = np.zeros((self.k_endog, self.nobs, n))
    partials_forecasts_error_cov = np.zeros((self.k_endog, self.k_endog, self.nobs, n))
    if approx_complex_step:
        epsilon = _get_epsilon(params, 2, None, n)
        increments = np.identity(n) * 1j * epsilon
        for i, ih in enumerate(increments):
            self.update(params + ih, transformed=transformed, includes_fixed=includes_fixed, complex_step=True)
            _res = self.ssm.filter(complex_step=True, **kwargs)
            partials_forecasts_error[:, :, i] = _res.forecasts_error.imag / epsilon[i]
            partials_forecasts_error_cov[:, :, :, i] = _res.forecasts_error_cov.imag / epsilon[i]
    elif not approx_centered:
        epsilon = _get_epsilon(params, 2, None, n)
        ei = np.zeros((n,), float)
        for i in range(n):
            ei[i] = epsilon[i]
            self.update(params + ei, transformed=transformed, includes_fixed=includes_fixed, complex_step=False)
            _res = self.ssm.filter(complex_step=False, **kwargs)
            partials_forecasts_error[:, :, i] = (_res.forecasts_error - res.forecasts_error) / epsilon[i]
            partials_forecasts_error_cov[:, :, :, i] = (_res.forecasts_error_cov - res.forecasts_error_cov) / epsilon[i]
            ei[i] = 0.0
    else:
        epsilon = _get_epsilon(params, 3, None, n) / 2.0
        ei = np.zeros((n,), float)
        for i in range(n):
            ei[i] = epsilon[i]
            self.update(params + ei, transformed=transformed, includes_fixed=includes_fixed, complex_step=False)
            _res1 = self.ssm.filter(complex_step=False, **kwargs)
            self.update(params - ei, transformed=transformed, includes_fixed=includes_fixed, complex_step=False)
            _res2 = self.ssm.filter(complex_step=False, **kwargs)
            partials_forecasts_error[:, :, i] = (_res1.forecasts_error - _res2.forecasts_error) / (2 * epsilon[i])
            partials_forecasts_error_cov[:, :, :, i] = (_res1.forecasts_error_cov - _res2.forecasts_error_cov) / (2 * epsilon[i])
            ei[i] = 0.0
    return (partials_forecasts_error, partials_forecasts_error_cov)