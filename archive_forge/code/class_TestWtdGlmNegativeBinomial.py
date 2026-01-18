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
class TestWtdGlmNegativeBinomial(CheckWtdDuplicationMixin):

    @classmethod
    def setup_class(cls):
        """
        Tests Negative Binomial family with canonical link
        g(p) = log(p/(p + 1/alpha))
        """
        super().setup_class()
        alpha = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DomainWarning)
            family_link = sm.families.NegativeBinomial(link=sm.families.links.NegativeBinomial(alpha=alpha), alpha=alpha)
            cls.res1 = GLM(cls.endog, cls.exog, freq_weights=cls.weight, family=family_link).fit()
            cls.res2 = GLM(cls.endog_big, cls.exog_big, family=family_link).fit()