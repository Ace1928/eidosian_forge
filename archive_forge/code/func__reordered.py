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
def _reordered(self, order):
    endog = self.endog
    endog_lagged = self.endog_lagged
    params = self.params
    sigma_u = self.sigma_u
    names = self.names
    k_ar = self.k_ar
    endog_new = np.zeros_like(endog)
    endog_lagged_new = np.zeros_like(endog_lagged)
    params_new_inc = np.zeros_like(params)
    params_new = np.zeros_like(params)
    sigma_u_new_inc = np.zeros_like(sigma_u)
    sigma_u_new = np.zeros_like(sigma_u)
    num_end = len(self.params[0])
    names_new = []
    k = self.k_trend
    for i, c in enumerate(order):
        endog_new[:, i] = self.endog[:, c]
        if k > 0:
            params_new_inc[0, i] = params[0, i]
            endog_lagged_new[:, 0] = endog_lagged[:, 0]
        for j in range(k_ar):
            params_new_inc[i + j * num_end + k, :] = self.params[c + j * num_end + k, :]
            endog_lagged_new[:, i + j * num_end + k] = endog_lagged[:, c + j * num_end + k]
        sigma_u_new_inc[i, :] = sigma_u[c, :]
        names_new.append(names[c])
    for i, c in enumerate(order):
        params_new[:, i] = params_new_inc[:, c]
        sigma_u_new[:, i] = sigma_u_new_inc[:, c]
    return VARResults(endog=endog_new, endog_lagged=endog_lagged_new, params=params_new, sigma_u=sigma_u_new, lag_order=self.k_ar, model=self.model, trend='c', names=names_new, dates=self.dates)