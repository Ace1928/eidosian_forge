import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.datasets import cpunish, longley
from statsmodels.discrete import discrete_model as discrete
from statsmodels.genmod.generalized_linear_model import GLM, SET_USE_BIC_LLF
from statsmodels.tools.numdiff import (
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import add_constant
class TestTweedieLog15Fair(CheckTweedie):

    @classmethod
    def setup_class(cls):
        from statsmodels.datasets.fair import load_pandas
        from .results.results_glm import FairTweedieLog15
        data = load_pandas()
        family_link = sm.families.Tweedie(link=sm.families.links.Log(), var_power=1.5)
        cls.res1 = sm.GLM(endog=data.endog, exog=data.exog[['rate_marriage', 'age', 'yrs_married']], family=family_link).fit()
        cls.res2 = FairTweedieLog15()