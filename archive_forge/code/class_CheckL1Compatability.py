from statsmodels.compat.pandas import assert_index_equal
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import nbinom
import statsmodels.api as sm
from statsmodels.discrete.discrete_margins import _iscount, _isdummy
from statsmodels.discrete.discrete_model import (
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import (
from .results.results_discrete import Anes, DiscreteL1, RandHIE, Spector
class CheckL1Compatability:
    """
    Tests compatability between l1 and unregularized by setting alpha such
    that certain parameters should be effectively unregularized, and others
    should be ignored by the model.
    """

    def test_params(self):
        m = self.m
        assert_almost_equal(self.res_unreg.params[:m], self.res_reg.params[:m], DECIMAL_4)
        kvars = self.res_reg.model.exog.shape[1]
        assert_almost_equal(0, self.res_reg.params[m:kvars], DECIMAL_4)

    def test_cov_params(self):
        m = self.m
        assert_almost_equal(self.res_unreg.cov_params()[:m, :m], self.res_reg.cov_params()[:m, :m], DECIMAL_1)

    def test_df(self):
        assert_equal(self.res_unreg.df_model, self.res_reg.df_model)
        assert_equal(self.res_unreg.df_resid, self.res_reg.df_resid)

    def test_t_test(self):
        m = self.m
        kvars = self.kvars
        extra = getattr(self, 'k_extra', 0)
        t_unreg = self.res_unreg.t_test(np.eye(len(self.res_unreg.params)))
        t_reg = self.res_reg.t_test(np.eye(kvars + extra))
        assert_almost_equal(t_unreg.effect[:m], t_reg.effect[:m], DECIMAL_3)
        assert_almost_equal(t_unreg.sd[:m], t_reg.sd[:m], DECIMAL_3)
        assert_almost_equal(np.nan, t_reg.sd[m])
        assert_allclose(t_unreg.tvalue[:m], t_reg.tvalue[:m], atol=0.003)
        assert_almost_equal(np.nan, t_reg.tvalue[m])

    def test_f_test(self):
        m = self.m
        kvars = self.kvars
        extra = getattr(self, 'k_extra', 0)
        f_unreg = self.res_unreg.f_test(np.eye(len(self.res_unreg.params))[:m])
        f_reg = self.res_reg.f_test(np.eye(kvars + extra)[:m])
        assert_allclose(f_unreg.fvalue, f_reg.fvalue, rtol=3e-05, atol=0.001)
        assert_almost_equal(f_unreg.pvalue, f_reg.pvalue, DECIMAL_3)

    def test_bad_r_matrix(self):
        kvars = self.kvars
        assert_raises(ValueError, self.res_reg.f_test, np.eye(kvars))