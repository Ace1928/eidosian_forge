from __future__ import annotations
from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import deprecate_kwarg
from statsmodels.compat.python import lzip
from statsmodels.compat.scipy import _next_regular
from typing import Literal, Union
import warnings
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import correlate
from statsmodels.regression.linear_model import OLS, yule_walker
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tools.validation import (
from statsmodels.tsa._bds import bds
from statsmodels.tsa._innovations import innovations_algo, innovations_filter
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
from statsmodels.tsa.tsatools import add_trend, lagmat, lagmat2ds
def _safe_arma_fit(y, order, model_kw, trend, fit_kw, start_params=None):
    from statsmodels.tsa.arima.model import ARIMA
    try:
        return ARIMA(y, order=order, **model_kw, trend=trend).fit(start_params=start_params, **fit_kw)
    except LinAlgError:
        return
    except ValueError as error:
        if start_params is not None:
            return
        elif 'initial' not in error.args[0] or 'initial' in str(error):
            start_params = [0.1] * sum(order)
            if trend == 'c':
                start_params = [0.1] + start_params
            return _safe_arma_fit(y, order, model_kw, trend, fit_kw, start_params)
        else:
            return
    except:
        return