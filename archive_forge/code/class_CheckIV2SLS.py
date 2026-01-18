from statsmodels.compat.python import lrange, lmap
import os
import copy
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
import statsmodels.sandbox.regression.gmm as gmm
class CheckIV2SLS:

    def test_basic(self):
        res1, res2 = (self.res1, self.res2)
        assert_allclose(res1.params, res2.params, rtol=1e-09, atol=0)
        assert_allclose(res1.params, res2.params, rtol=0, atol=1e-10)
        n = res1.model.exog.shape[0]
        assert_allclose(res1.bse, res2.bse, rtol=1e-10, atol=0)
        assert_allclose(res1.bse, res2.bse, rtol=0, atol=1e-11)
        assert_allclose(res1.tvalues, res2.tvalues, rtol=5e-10, atol=0)

    def test_other(self):
        res1, res2 = (self.res1, self.res2)
        assert_allclose(res1.rsquared, res2.r2, rtol=1e-07, atol=0)
        assert_allclose(res1.rsquared_adj, res2.r2_a, rtol=1e-07, atol=0)
        assert_allclose(res1.fvalue, res2.F, rtol=1e-10, atol=0)
        assert_allclose(res1.f_pvalue, res2.Fp, rtol=1e-08, atol=0)
        assert_allclose(np.sqrt(res1.mse_resid), res2.rmse, rtol=1e-10, atol=0)
        assert_allclose(res1.ssr, res2.rss, rtol=1e-10, atol=0)
        assert_allclose(res1.uncentered_tss, res2.yy, rtol=1e-10, atol=0)
        assert_allclose(res1.centered_tss, res2.yyc, rtol=1e-10, atol=0)
        assert_allclose(res1.ess, res2.mss, rtol=1e-09, atol=0)
        assert_equal(res1.df_model, res2.df_m)
        assert_equal(res1.df_resid, res2.df_r)

    def test_hypothesis(self):
        res1, res2 = (self.res1, self.res2)
        restriction = np.eye(len(res1.params))
        res_t = res1.t_test(restriction)
        assert_allclose(res_t.tvalue, res1.tvalues, rtol=1e-12, atol=0)
        assert_allclose(res_t.pvalue, res1.pvalues, rtol=1e-12, atol=0)
        res_f = res1.f_test(restriction[:-1])
        assert_allclose(res_f.fvalue, res1.fvalue, rtol=1e-12, atol=0)
        assert_allclose(res_f.pvalue, res1.f_pvalue, rtol=1e-10, atol=0)
        assert_allclose(res_f.fvalue, res2.F, rtol=1e-10, atol=0)
        assert_allclose(res_f.pvalue, res2.Fp, rtol=1e-08, atol=0)

    def test_hausman(self):
        res1, res2 = (self.res1, self.res2)
        hausm = res1.spec_hausman()
        assert_allclose(hausm[0], res2.hausman['DWH'], rtol=1e-11, atol=0)
        assert_allclose(hausm[1], res2.hausman['DWHp'], rtol=1e-10, atol=1e-25)

    @pytest.mark.smoke
    def test_summary(self):
        res1 = self.res1
        summ = res1.summary()
        assert_equal(len(summ.tables[1]), len(res1.params) + 1)