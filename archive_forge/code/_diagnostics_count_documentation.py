import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.stats.base import HolderTuple
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.regression.linear_model import OLS
Test for excess zeros in Poisson regression model.

    The test is implemented following Tang and Tang [1]_ equ. (12) which is
    based on the test derived in He et al 2019 [2]_.

    References
    ----------

    .. [1] Tang, Yi, and Wan Tang. 2018. “Testing Modified Zeros for Poisson
           Regression Models:” Statistical Methods in Medical Research,
           September. https://doi.org/10.1177/0962280218796253.

    .. [2] He, Hua, Hui Zhang, Peng Ye, and Wan Tang. 2019. “A Test of Inflated
           Zeros for Poisson Regression Models.” Statistical Methods in
           Medical Research 28 (4): 1157–69.
           https://doi.org/10.1177/0962280217749991.

    