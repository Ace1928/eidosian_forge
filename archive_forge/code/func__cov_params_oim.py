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
def _cov_params_oim(self, approx_complex_step=True, approx_centered=False):
    evaluated_hessian = self.nobs_effective * self.model.hessian(self.params, hessian_method='oim', transformed=True, includes_fixed=True, approx_complex_step=approx_complex_step, approx_centered=approx_centered)
    if len(self.fixed_params) > 0:
        mask = np.ix_(self._free_params_index, self._free_params_index)
        tmp, singular_values = pinv_extended(evaluated_hessian[mask])
        neg_cov = np.zeros_like(evaluated_hessian) * np.nan
        neg_cov[mask] = tmp
    else:
        neg_cov, singular_values = pinv_extended(evaluated_hessian)
    self.model.update(self.params, transformed=True, includes_fixed=True)
    if self._rank is None:
        self._rank = np.linalg.matrix_rank(np.diag(singular_values))
    return -neg_cov