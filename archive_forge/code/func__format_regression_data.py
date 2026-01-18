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
def _format_regression_data(self, series, nobs, const, trend, cols, lags):
    """
        Create the endog/exog data for the auxiliary regressions
        from the original (standardized) series under test.
        """
    endog = np.diff(series, axis=0)
    endog /= np.sqrt(endog.T.dot(endog))
    series /= np.sqrt(series.T.dot(series))
    exog = np.zeros((endog[lags:].shape[0], cols + lags))
    exog[:, 0] = const
    exog[:, cols - 1] = series[lags:nobs - 1]
    exog[:, cols:] = lagmat(endog, lags, trim='none')[lags:exog.shape[0] + lags]
    return (endog, exog)