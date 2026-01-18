from statsmodels.compat.python import lmap
import os
import numpy as np
from scipy import stats
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
from .try_ols_anova import data2dummy
calculating anova and verifying with NIST test data

compares my implementations, stats.f_oneway and anova using statsmodels.OLS
