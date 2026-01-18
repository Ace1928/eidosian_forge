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

    Cusum test for parameter stability based on ols residuals.

    Parameters
    ----------
    resid : ndarray
        An array of residuals from an OLS estimation.
    ddof : int
        The number of parameters in the OLS estimation, used as degrees
        of freedom correction for error variance.

    Returns
    -------
    sup_b : float
        The test statistic, maximum of absolute value of scaled cumulative OLS
        residuals.
    pval : float
        Probability of observing the data under the null hypothesis of no
        structural change, based on asymptotic distribution which is a Brownian
        Bridge
    crit: list
        The tabulated critical values, for alpha = 1%, 5% and 10%.

    Notes
    -----
    Tested against R:structchange.

    Not clear: Assumption 2 in Ploberger, Kramer assumes that exog x have
    asymptotically zero mean, x.mean(0) = [1, 0, 0, ..., 0]
    Is this really necessary? I do not see how it can affect the test statistic
    under the null. It does make a difference under the alternative.
    Also, the asymptotic distribution of test statistic depends on this.

    From examples it looks like there is little power for standard cusum if
    exog (other than constant) have mean zero.

    References
    ----------
    Ploberger, Werner, and Walter Kramer. “The Cusum Test with OLS Residuals.”
    Econometrica 60, no. 2 (March 1992): 271-285.
    