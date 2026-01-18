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
def _check_nested_exog(small, large):
    """
    Check if a larger exog nests a smaller exog

    Parameters
    ----------
    small : ndarray
        exog from smaller model
    large : ndarray
        exog from larger model

    Returns
    -------
    bool
        True if small is nested by large
    """
    if small.shape[1] > large.shape[1]:
        return False
    coef = np.linalg.lstsq(large, small, rcond=None)[0]
    err = small - large @ coef
    return np.linalg.matrix_rank(np.c_[large, err]) == large.shape[1]