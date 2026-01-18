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
def _quick_ols(self, endog, exog):
    """
        Minimal implementation of LS estimator for internal use
        """
    xpxi = np.linalg.inv(exog.T.dot(exog))
    xpy = exog.T.dot(endog)
    nobs, k_exog = exog.shape
    b = xpxi.dot(xpy)
    e = endog - exog.dot(b)
    sigma2 = e.T.dot(e) / (nobs - k_exog)
    return b / np.sqrt(np.diag(sigma2 * xpxi))