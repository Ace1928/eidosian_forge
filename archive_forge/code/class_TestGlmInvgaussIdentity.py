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
class TestGlmInvgaussIdentity(CheckModelResultsMixin):

    @classmethod
    def setup_class(cls):
        cls.decimal_aic_R = -10
        cls.decimal_fittedvalues = DECIMAL_3
        cls.decimal_params = DECIMAL_3
        from .results.results_glm import Medpar1
        data = Medpar1()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cls.res1 = GLM(data.endog, data.exog, family=sm.families.InverseGaussian(link=sm.families.links.Identity())).fit()
        from .results.results_glm import InvGaussIdentity
        cls.res2 = InvGaussIdentity()