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
def _handle_args(names, defaults, *args, **kwargs):
    output_args = []
    if len(args) > 0:
        if isinstance(args[0], dict):
            flags = args[0]
        else:
            flags = dict(zip(names, args))
        for i in range(len(names)):
            output_args.append(flags.get(names[i], defaults[i]))
        for name, value in flags.items():
            if name in kwargs:
                raise TypeError("loglike() got multiple values for keyword argument '%s'" % name)
    else:
        for i in range(len(names)):
            output_args.append(kwargs.pop(names[i], defaults[i]))
    return tuple(output_args) + (kwargs,)