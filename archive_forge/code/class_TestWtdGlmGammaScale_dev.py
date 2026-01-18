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
class TestWtdGlmGammaScale_dev(CheckWtdDuplicationMixin):

    @classmethod
    def setup_class(cls):
        """
        Tests Gamma family with log link.
        """
        super().setup_class()
        family_link = sm.families.Gamma(sm.families.links.Log())
        cls.res1 = GLM(cls.endog, cls.exog, freq_weights=cls.weight, family=family_link).fit(scale='dev')
        cls.res2 = GLM(cls.endog_big, cls.exog_big, family=family_link).fit(scale='dev')

    def test_missing(self):
        endog = self.data.endog.copy()
        exog = self.data.exog.copy()
        exog[0, 0] = np.nan
        endog[[2, 4, 6, 8]] = np.nan
        freq_weights = self.weight
        mod_misisng = GLM(endog, exog, family=self.res1.model.family, freq_weights=freq_weights, missing='drop')
        assert_equal(mod_misisng.freq_weights.shape[0], mod_misisng.endog.shape[0])
        assert_equal(mod_misisng.freq_weights.shape[0], mod_misisng.exog.shape[0])
        keep_idx = np.array([1, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16])
        assert_equal(mod_misisng.freq_weights, self.weight[keep_idx])