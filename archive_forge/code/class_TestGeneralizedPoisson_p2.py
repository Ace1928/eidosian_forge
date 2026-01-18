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
class TestGeneralizedPoisson_p2:

    @classmethod
    def setup_class(cls):
        data = load_randhie()
        data.exog = sm.add_constant(data.exog, prepend=False)
        mod = GeneralizedPoisson(data.endog, data.exog, p=2)
        cls.res1 = mod.fit(method='newton', disp=0)
        res2 = RandHIE.generalizedpoisson_gp2
        cls.res2 = res2

    def test_bse(self):
        assert_allclose(self.res1.bse, self.res2.bse, atol=1e-05)

    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params, atol=1e-05)

    def test_alpha(self):
        assert_allclose(self.res1.lnalpha, self.res2.lnalpha)
        assert_allclose(self.res1.lnalpha_std_err, self.res2.lnalpha_std_err, atol=1e-05)

    def test_conf_int(self):
        assert_allclose(self.res1.conf_int(), self.res2.conf_int, atol=0.001)

    def test_aic(self):
        assert_allclose(self.res1.aic, self.res2.aic)

    def test_bic(self):
        assert_allclose(self.res1.bic, self.res2.bic)

    def test_df(self):
        assert_equal(self.res1.df_model, self.res2.df_model)

    def test_llf(self):
        assert_allclose(self.res1.llf, self.res2.llf)

    def test_wald(self):
        result = self.res1.wald_test(np.eye(len(self.res1.params))[:-2], scalar=True)
        assert_allclose(result.statistic, self.res2.wald_statistic)
        assert_allclose(result.pvalue, self.res2.wald_pvalue, atol=1e-15)

    def test_t(self):
        unit_matrix = np.identity(self.res1.params.size)
        t_test = self.res1.t_test(unit_matrix)
        assert_allclose(self.res1.tvalues, t_test.tvalue)

    def test_jac(self):
        check_jac(self)

    def test_distr(self):
        check_distr(self.res1)