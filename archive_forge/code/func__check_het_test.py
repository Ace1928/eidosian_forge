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
def _check_het_test(x: np.ndarray, test_name: str) -> None:
    """
    Check validity of the exogenous regressors in a heteroskedasticity test

    Parameters
    ----------
    x : ndarray
        The exogenous regressor array
    test_name : str
        The test name for the exception
    """
    x_max = x.max(axis=0)
    if not np.any((x_max - x.min(axis=0) == 0) & (x_max != 0)) or x.shape[1] < 2:
        raise ValueError(f'{test_name} test requires exog to have at least two columns where one is a constant.')