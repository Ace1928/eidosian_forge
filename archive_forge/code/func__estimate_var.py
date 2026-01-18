from __future__ import annotations
from statsmodels.compat.python import lrange
from collections import defaultdict
from io import StringIO
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly, deprecated_alias
from statsmodels.tools.linalg import logdet_symm
from statsmodels.tools.sm_exceptions import OutputWarning
from statsmodels.tools.validation import array_like
from statsmodels.tsa.base.tsa_model import (
import statsmodels.tsa.tsatools as tsa
from statsmodels.tsa.tsatools import duplication_matrix, unvec, vec
from statsmodels.tsa.vector_ar import output, plotting, util
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.output import VARSummary
def _estimate_var(self, lags, offset=0, trend='c'):
    """
        lags : int
            Lags of the endogenous variable.
        offset : int
            Periods to drop from beginning-- for order selection so it's an
            apples-to-apples comparison
        trend : {str, None}
            As per above
        """
    self.k_trend = k_trend = util.get_trendorder(trend)
    if offset < 0:
        raise ValueError('offset must be >= 0')
    nobs = self.n_totobs - lags - offset
    endog = self.endog[offset:]
    exog = None if self.exog is None else self.exog[offset:]
    z = util.get_var_endog(endog, lags, trend=trend, has_constant='raise')
    if exog is not None:
        x = util.get_var_endog(exog[-nobs:], 0, trend='n', has_constant='raise')
        x_inst = exog[-nobs:]
        x = np.column_stack((x, x_inst))
        del x_inst
        temp_z = z
        z = np.empty((x.shape[0], x.shape[1] + z.shape[1]))
        z[:, :self.k_trend] = temp_z[:, :self.k_trend]
        z[:, self.k_trend:self.k_trend + x.shape[1]] = x
        z[:, self.k_trend + x.shape[1]:] = temp_z[:, self.k_trend:]
        del temp_z, x
    for i in range(self.k_trend):
        if (np.diff(z[:, i]) == 1).all():
            z[:, i] += lags
        if (np.diff(np.sqrt(z[:, i])) == 1).all():
            z[:, i] = (np.sqrt(z[:, i]) + lags) ** 2
    y_sample = endog[lags:]
    params = np.linalg.lstsq(z, y_sample, rcond=1e-15)[0]
    resid = y_sample - np.dot(z, params)
    avobs = len(y_sample)
    if exog is not None:
        k_trend += exog.shape[1]
    df_resid = avobs - (self.neqs * lags + k_trend)
    sse = np.dot(resid.T, resid)
    if df_resid:
        omega = sse / df_resid
    else:
        omega = np.full_like(sse, np.nan)
    varfit = VARResults(endog, z, params, omega, lags, names=self.endog_names, trend=trend, dates=self.data.dates, model=self, exog=self.exog)
    return VARResultsWrapper(varfit)