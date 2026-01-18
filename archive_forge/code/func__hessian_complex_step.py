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
def _hessian_complex_step(self, params, **kwargs):
    """
        Hessian matrix computed by second-order complex-step differentiation
        on the `loglike` function.
        """
    epsilon = _get_epsilon(params, 3.0, None, len(params))
    kwargs['transformed'] = True
    kwargs['complex_step'] = True
    hessian = approx_hess_cs(params, self.loglike, epsilon=epsilon, kwargs=kwargs)
    return hessian / (self.nobs - self.ssm.loglikelihood_burn)