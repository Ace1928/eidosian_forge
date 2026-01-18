from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from pandas import MultiIndex, get_dummies
from scipy import special, stats
from scipy.special import digamma, gammaln, loggamma, polygamma
from scipy.stats import nbinom
from statsmodels.base.data import handle_data  # for mnlogit
from statsmodels.base.l1_slsqp import fit_l1_slsqp
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.base._constraints import fit_constrained_wrap
import statsmodels.base._parameter_inference as pinfer
from statsmodels.base import _prediction_inference as pred
from statsmodels.distributions import genpoisson_p
import statsmodels.regression.linear_model as lm
from statsmodels.tools import data as data_tools, tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tools.sm_exceptions import (
def _numpy_to_dummies(endog):
    if endog.ndim == 2 and endog.dtype.kind not in ['S', 'O']:
        endog_dummies = endog
        ynames = range(endog.shape[1])
    else:
        dummies = get_dummies(endog, drop_first=False)
        ynames = {i: dummies.columns[i] for i in range(dummies.shape[1])}
        endog_dummies = np.asarray(dummies, dtype=float)
        return (endog_dummies, ynames)
    return (endog_dummies, ynames)