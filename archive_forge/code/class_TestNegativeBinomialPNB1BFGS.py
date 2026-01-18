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
class TestNegativeBinomialPNB1BFGS(CheckNegBinMixin, CheckModelResults):

    @classmethod
    def setup_class(cls):
        data = load_randhie()
        exog = sm.add_constant(data.exog, prepend=False)
        cls.res1 = NegativeBinomialP(data.endog, exog, p=1).fit(method='bfgs', maxiter=100, disp=0)
        res2 = RandHIE.negativebinomial_nb1_bfgs
        cls.res2 = res2

    def test_bse(self):
        assert_allclose(self.res1.bse, self.res2.bse, atol=0.005, rtol=0.005)

    def test_aic(self):
        assert_allclose(self.res1.aic, self.res2.aic, atol=0.5, rtol=0.5)

    def test_bic(self):
        assert_allclose(self.res1.bic, self.res2.bic, atol=0.5, rtol=0.5)

    def test_llf(self):
        assert_allclose(self.res1.llf, self.res2.llf, atol=0.001, rtol=0.001)

    def test_llr(self):
        assert_allclose(self.res1.llf, self.res2.llf, atol=0.001, rtol=0.001)

    def test_zstat(self):
        assert_allclose(self.res1.tvalues, self.res2.z, atol=0.5, rtol=0.5)

    def test_lnalpha(self):
        assert_allclose(self.res1.lnalpha, self.res2.lnalpha, atol=0.001, rtol=0.001)
        assert_allclose(self.res1.lnalpha_std_err, self.res2.lnalpha_std_err, atol=0.001, rtol=0.001)

    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params, atol=0.05, rtol=0.05)

    def test_conf_int(self):
        assert_allclose(self.res1.conf_int(), self.res2.conf_int, atol=0.05, rtol=0.05)

    def test_predict(self):
        assert_allclose(self.res1.predict()[:10], np.exp(self.res2.fittedvalues[:10]), atol=0.005, rtol=0.005)

    def test_predict_xb(self):
        assert_allclose(self.res1.predict(which='linear')[:10], self.res2.fittedvalues[:10], atol=0.005, rtol=0.005)

    def test_init_kwds(self):
        kwds = self.res1.model._get_init_kwds()
        assert_('p' in kwds)
        assert_equal(kwds['p'], 1)