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
def _hessian_finite_difference(self, params, approx_centered=False, **kwargs):
    params = np.array(params, ndmin=1)
    warnings.warn('Calculation of the Hessian using finite differences is usually subject to substantial approximation errors.', PrecisionWarning)
    if not approx_centered:
        epsilon = _get_epsilon(params, 3, None, len(params))
    else:
        epsilon = _get_epsilon(params, 4, None, len(params)) / 2
    hessian = approx_fprime(params, self._score_finite_difference, epsilon=epsilon, kwargs=kwargs, centered=approx_centered)
    return hessian / (self.nobs - self.ssm.loglikelihood_burn)