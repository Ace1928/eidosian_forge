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
class TestGlmPoisson(CheckModelResultsMixin, CheckComparisonMixin):

    @classmethod
    def setup_class(cls):
        """
        Tests Poisson family with canonical log link.

        Test results were obtained by R.
        """
        from .results.results_glm import Cpunish
        cls.data = cpunish.load()
        cls.data.endog = np.require(cls.data.endog, requirements='W')
        cls.data.exog = np.require(cls.data.exog, requirements='W')
        cls.data.exog[:, 3] = np.log(cls.data.exog[:, 3])
        cls.data.exog = add_constant(cls.data.exog, prepend=False)
        cls.res1 = GLM(cls.data.endog, cls.data.exog, family=sm.families.Poisson()).fit()
        cls.res2 = Cpunish()
        modd = discrete.Poisson(cls.data.endog, cls.data.exog)
        cls.resd = modd.fit(start_params=cls.res1.params * 0.9, disp=False)