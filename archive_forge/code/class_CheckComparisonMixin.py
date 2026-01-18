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
class CheckComparisonMixin:

    def test_compare_discrete(self):
        res1 = self.res1
        resd = self.resd
        assert_allclose(res1.llf, resd.llf, rtol=1e-10)
        score_obs1 = res1.model.score_obs(res1.params * 0.98)
        score_obsd = resd.model.score_obs(resd.params * 0.98)
        assert_allclose(score_obs1, score_obsd, rtol=1e-10)
        score1 = res1.model.score(res1.params * 0.98)
        assert_allclose(score1, score_obs1.sum(0), atol=1e-20)
        score0 = res1.model.score(res1.params)
        assert_allclose(score0, np.zeros(score_obs1.shape[1]), atol=5e-07)
        hessian1 = res1.model.hessian(res1.params * 0.98, observed=False)
        hessiand = resd.model.hessian(resd.params * 0.98)
        assert_allclose(hessian1, hessiand, rtol=1e-10)
        hessian1 = res1.model.hessian(res1.params * 0.98, observed=True)
        hessiand = resd.model.hessian(resd.params * 0.98)
        assert_allclose(hessian1, hessiand, rtol=1e-09)

    def test_score_test(self):
        res1 = self.res1
        st, pv, df = res1.model.score_test(res1.params, k_constraints=1)
        assert_allclose(st, 0, atol=1e-20)
        assert_allclose(pv, 1, atol=1e-10)
        assert_equal(df, 1)
        st, pv, df = res1.model.score_test(res1.params, k_constraints=0)
        assert_allclose(st, 0, atol=1e-20)
        assert_(np.isnan(pv), msg=repr(pv))
        assert_equal(df, 0)
        exog_extra = res1.model.exog[:, 1] ** 2
        st, pv, df = res1.model.score_test(res1.params, exog_extra=exog_extra)
        assert_array_less(0.1, st)
        assert_array_less(0.1, pv)
        assert_equal(df, 1)

    def test_get_prediction(self):
        pred1 = self.res1.get_prediction()
        predd = self.resd.get_prediction()
        assert_allclose(predd.predicted, pred1.predicted_mean, rtol=1e-11)
        assert_allclose(predd.se, pred1.se_mean, rtol=1e-06)
        assert_allclose(predd.summary_frame().values, pred1.summary_frame().values, rtol=1e-06)
        pred1 = self.res1.get_prediction(which='mean')
        predd = self.resd.get_prediction()
        assert_allclose(predd.predicted, pred1.predicted, rtol=1e-11)
        assert_allclose(predd.se, pred1.se, rtol=1e-06)
        assert_allclose(predd.summary_frame().values, pred1.summary_frame().values, rtol=1e-06)