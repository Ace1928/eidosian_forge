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
class TestGlmGaussian(CheckModelResultsMixin):

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
        cls.data.endog = np.require(cls.data.endog, requirements='W')
        cls.data.exog = np.require(cls.data.exog, requirements='W')
        cls.data.exog = add_constant(cls.data.exog, prepend=False)
        cls.res1 = GLM(cls.data.endog, cls.data.exog, family=sm.families.Gaussian()).fit()
        from .results.results_glm import Longley
        cls.res2 = Longley()

    def test_compare_OLS(self):
        res1 = self.res1
        from statsmodels.regression.linear_model import OLS
        resd = OLS(self.data.endog, self.data.exog).fit(use_t=False)
        self.resd = resd
        assert_allclose(res1.llf, resd.llf, rtol=1e-10)
        score_obs1 = res1.model.score_obs(res1.params, scale=None)
        score_obsd = resd.resid[:, None] / resd.scale * resd.model.exog
        assert_allclose(score_obs1, score_obsd, rtol=1e-08)
        score_obs1 = res1.model.score_obs(res1.params, scale=1)
        score_obsd = resd.resid[:, None] * resd.model.exog
        assert_allclose(score_obs1, score_obsd, rtol=1e-08)
        hess_obs1 = res1.model.hessian(res1.params, scale=None)
        hess_obsd = -1.0 / resd.scale * resd.model.exog.T.dot(resd.model.exog)
        assert_allclose(hess_obs1, hess_obsd, rtol=1e-08)
        pred1 = res1.get_prediction()
        predd = resd.get_prediction()
        assert_allclose(predd.predicted, pred1.predicted_mean, rtol=1e-11)
        assert_allclose(predd.se, pred1.se_mean, rtol=1e-06)
        assert_allclose(predd.summary_frame().values[:, :4], pred1.summary_frame().values, rtol=1e-06)
        pred1 = self.res1.get_prediction(which='mean')
        predd = self.resd.get_prediction()
        assert_allclose(predd.predicted, pred1.predicted, rtol=1e-11)
        assert_allclose(predd.se, pred1.se, rtol=1e-06)
        assert_allclose(predd.summary_frame().values[:, :4], pred1.summary_frame().values, rtol=1e-06)