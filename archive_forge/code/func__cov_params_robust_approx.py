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
def _cov_params_robust_approx(self, approx_complex_step=True, approx_centered=False):
    cov_opg = self._cov_params_opg(approx_complex_step=approx_complex_step, approx_centered=approx_centered)
    evaluated_hessian = self.nobs_effective * self.model.hessian(self.params, transformed=True, includes_fixed=True, method='approx', approx_complex_step=approx_complex_step)
    if len(self.fixed_params) > 0:
        mask = np.ix_(self._free_params_index, self._free_params_index)
        cov_params = np.zeros_like(evaluated_hessian) * np.nan
        cov_opg = cov_opg[mask]
        evaluated_hessian = evaluated_hessian[mask]
        tmp, singular_values = pinv_extended(np.dot(np.dot(evaluated_hessian, cov_opg), evaluated_hessian))
        cov_params[mask] = tmp
    else:
        cov_params, singular_values = pinv_extended(np.dot(np.dot(evaluated_hessian, cov_opg), evaluated_hessian))
    self.model.update(self.params, transformed=True, includes_fixed=True)
    if self._rank is None:
        self._rank = np.linalg.matrix_rank(np.diag(singular_values))
    return cov_params