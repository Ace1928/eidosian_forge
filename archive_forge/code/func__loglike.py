from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import (
from collections import namedtuple
import numpy as np
from pandas import DataFrame, MultiIndex, Series
from scipy import stats
from statsmodels.base import model
from statsmodels.base.model import LikelihoodModelResults, Model
from statsmodels.regression.linear_model import (
from statsmodels.tools.validation import array_like, int_like, string_like
def _loglike(self, params, wy, wx, weights, nobs):
    nobs2 = nobs / 2.0
    wresid = wy - wx @ params
    ssr = np.sum(wresid ** 2, axis=0)
    llf = -np.log(ssr) * nobs2
    llf -= (1 + np.log(np.pi / nobs2)) * nobs2
    llf += 0.5 * np.sum(np.log(weights))
    return (wresid, ssr, llf)