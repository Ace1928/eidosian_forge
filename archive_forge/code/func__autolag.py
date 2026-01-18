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
def _autolag(mod, endog, exog, startlag, maxlag, method, modargs=(), fitargs=(), regresults=False):
    """
    Returns the results for the lag length that maximizes the info criterion.

    Parameters
    ----------
    mod : Model class
        Model estimator class
    endog : array_like
        nobs array containing endogenous variable
    exog : array_like
        nobs by (startlag + maxlag) array containing lags and possibly other
        variables
    startlag : int
        The first zero-indexed column to hold a lag.  See Notes.
    maxlag : int
        The highest lag order for lag length selection.
    method : {"aic", "bic", "t-stat"}
        aic - Akaike Information Criterion
        bic - Bayes Information Criterion
        t-stat - Based on last lag
    modargs : tuple, optional
        args to pass to model.  See notes.
    fitargs : tuple, optional
        args to pass to fit.  See notes.
    regresults : bool, optional
        Flag indicating to return optional return results

    Returns
    -------
    icbest : float
        Best information criteria.
    bestlag : int
        The lag length that maximizes the information criterion.
    results : dict, optional
        Dictionary containing all estimation results

    Notes
    -----
    Does estimation like mod(endog, exog[:,:i], *modargs).fit(*fitargs)
    where i goes from lagstart to lagstart+maxlag+1.  Therefore, lags are
    assumed to be in contiguous columns from low to high lag length with
    the highest lag in the last column.
    """
    results = {}
    method = method.lower()
    for lag in range(startlag, startlag + maxlag + 1):
        mod_instance = mod(endog, exog[:, :lag], *modargs)
        results[lag] = mod_instance.fit()
    if method == 'aic':
        icbest, bestlag = min(((v.aic, k) for k, v in results.items()))
    elif method == 'bic':
        icbest, bestlag = min(((v.bic, k) for k, v in results.items()))
    elif method == 't-stat':
        stop = 1.6448536269514722
        bestlag = startlag + maxlag
        icbest = 0.0
        for lag in range(startlag + maxlag, startlag - 1, -1):
            icbest = np.abs(results[lag].tvalues[-1])
            bestlag = lag
            if np.abs(icbest) >= stop:
                break
    else:
        raise ValueError(f'Information Criterion {method} not understood.')
    if not regresults:
        return (icbest, bestlag)
    else:
        return (icbest, bestlag, results)