from collections import OrderedDict
import contextlib
import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import norm, rv_continuous, rv_discrete
from scipy.stats.distributions import rv_frozen
from statsmodels.base.covtype import descriptions
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import forg
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import (
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.exponential_smoothing import base
import statsmodels.tsa.exponential_smoothing._ets_smooth as smooth
from statsmodels.tsa.exponential_smoothing.initialization import (
from statsmodels.tsa.tsatools import freq_to_period
def _loglike_internal(self, params, yhat, xhat, is_fixed=None, fixed_values=None, use_beta_star=False, use_gamma_star=False):
    """
        Log-likelihood function to be called from fit to avoid reallocation of
        memory.

        Parameters
        ----------
        params : np.ndarray of np.float
            Model parameters: (alpha, beta, gamma, phi, l[-1],
            b[-1], s[-1], ..., s[-m]). If there are no fixed values this must
            be in the format of internal parameters. Otherwise the fixed values
            are skipped.
        yhat : np.ndarray
            Array of size (n,) where fitted values will be written to.
        xhat : np.ndarray
            Array of size (n, _k_states_internal) where fitted states will be
            written to.
        is_fixed : np.ndarray or None
            Boolean array indicating values which are fixed during fitting.
            This must have the full length of internal parameters.
        fixed_values : np.ndarray or None
            Array of fixed values (arbitrary values for non-fixed parameters)
            This must have the full length of internal parameters.
        use_beta_star : boolean
            Whether to internally use beta_star as parameter
        use_gamma_star : boolean
            Whether to internally use gamma_star as parameter
        """
    if np.iscomplexobj(params):
        data = np.asarray(self.endog, dtype=complex)
    else:
        data = self.endog
    if is_fixed is None:
        is_fixed = np.zeros(self._k_params_internal, dtype=np.int64)
        fixed_values = np.empty(self._k_params_internal, dtype=params.dtype)
    else:
        is_fixed = np.ascontiguousarray(is_fixed, dtype=np.int64)
    self._smoothing_func(params, data, yhat, xhat, is_fixed, fixed_values, use_beta_star, use_gamma_star)
    res = self._residuals(yhat, data=data)
    logL = -self.nobs / 2 * (np.log(2 * np.pi * np.mean(res ** 2)) + 1)
    if self.error == 'mul':
        yhat[yhat <= 0] = 1 / (1e-08 * (1 + np.abs(yhat[yhat <= 0])))
        logL -= np.sum(np.log(yhat))
    return logL