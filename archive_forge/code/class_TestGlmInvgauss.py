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
class TestGlmInvgauss(CheckModelResultsMixin):

    @classmethod
    def setup_class(cls):
        """
        Tests the Inverse Gaussian family in GLM.

        Notes
        -----
        Used the rndivgx.ado file provided by Hardin and Hilbe to
        generate the data.  Results are read from model_results, which
        were obtained by running R_ig.s
        """
        cls.decimal_aic_R = DECIMAL_0
        cls.decimal_loglike = DECIMAL_0
        from .results.results_glm import InvGauss
        res2 = InvGauss()
        res1 = GLM(res2.endog, res2.exog, family=sm.families.InverseGaussian()).fit()
        cls.res1 = res1
        cls.res2 = res2

    def test_get_distribution(self):
        res1 = self.res1
        distr = res1.model.family.get_distribution(res1.fittedvalues, res1.scale)
        var_endog = res1.model.family.variance(res1.fittedvalues) * res1.scale
        m, v = distr.stats()
        assert_allclose(res1.fittedvalues, m, rtol=1e-13)
        assert_allclose(var_endog, v, rtol=1e-13)