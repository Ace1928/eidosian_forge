from statsmodels.compat.pandas import deprecate_kwarg
from collections.abc import Iterable
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper
from statsmodels.stats._adnorm import anderson_statistic, normal_ad
from statsmodels.stats._lilliefors import (
from statsmodels.tools.validation import (
from statsmodels.tsa.tsatools import lagmat
def _test_nested(endog, a, b, cov_est, cov_kwds):
    err = b - a @ np.linalg.lstsq(a, b, rcond=None)[0]
    u, s, v = np.linalg.svd(err)
    eps = np.finfo(np.double).eps
    tol = s.max(axis=-1, keepdims=True) * max(err.shape) * eps
    non_zero = np.abs(s) > tol
    aug = err @ v[:, non_zero]
    aug_reg = np.hstack([a, aug])
    k_a = aug.shape[1]
    k = aug_reg.shape[1]
    res = OLS(endog, aug_reg).fit(cov_type=cov_est, cov_kwds=cov_kwds)
    r_matrix = np.zeros((k_a, k))
    r_matrix[:, -k_a:] = np.eye(k_a)
    test = res.wald_test(r_matrix, use_f=True, scalar=True)
    stat, pvalue = (test.statistic, test.pvalue)
    df_num, df_denom = (int(test.df_num), int(test.df_denom))
    return (stat, pvalue, df_num, df_denom)