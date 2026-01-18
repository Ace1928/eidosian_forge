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
class TestGlmGammaIdentity(CheckModelResultsMixin):

    @classmethod
    def setup_class(cls):
        cls.decimal_resids = -100
        cls.decimal_params = DECIMAL_2
        cls.decimal_aic_R = DECIMAL_0
        cls.decimal_loglike = DECIMAL_1
        from .results.results_glm import CancerIdentity
        res2 = CancerIdentity()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fam = sm.families.Gamma(link=sm.families.links.Identity())
            cls.res1 = GLM(res2.endog, res2.exog, family=fam).fit()
        cls.res2 = res2