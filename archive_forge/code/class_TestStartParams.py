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
class TestStartParams(CheckModelResultsMixin):

    @classmethod
    def setup_class(cls):
        """
        Test Gaussian family with canonical identity link
        """
        cls.decimal_resids = DECIMAL_3
        cls.decimal_params = DECIMAL_2
        cls.decimal_bic = DECIMAL_0
        cls.decimal_bse = DECIMAL_3
        from statsmodels.datasets.longley import load
        cls.data = load()
        cls.data.exog = add_constant(cls.data.exog, prepend=False)
        params = sm.OLS(cls.data.endog, cls.data.exog).fit().params
        cls.res1 = GLM(cls.data.endog, cls.data.exog, family=sm.families.Gaussian()).fit(start_params=params)
        from .results.results_glm import Longley
        cls.res2 = Longley()