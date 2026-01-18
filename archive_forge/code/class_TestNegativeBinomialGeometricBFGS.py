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
class TestNegativeBinomialGeometricBFGS(CheckNegBinMixin, CheckModelResults):

    @classmethod
    def setup_class(cls):
        data = load_randhie()
        exog = sm.add_constant(data.exog, prepend=False)
        mod = NegativeBinomial(data.endog, exog, 'geometric')
        cls.res1 = mod.fit(method='bfgs', disp=0)
        res2 = RandHIE.negativebinomial_geometric_bfgs
        cls.res2 = res2

    def test_aic(self):
        assert_almost_equal(self.res1.aic, self.res2.aic, DECIMAL_3)

    def test_bic(self):
        assert_almost_equal(self.res1.bic, self.res2.bic, DECIMAL_3)

    def test_conf_int(self):
        assert_almost_equal(self.res1.conf_int(), self.res2.conf_int, DECIMAL_3)

    def test_fittedvalues(self):
        assert_almost_equal(self.res1.fittedvalues[:10], self.res2.fittedvalues[:10], DECIMAL_3)

    def test_predict(self):
        assert_almost_equal(self.res1.predict()[:10], np.exp(self.res2.fittedvalues[:10]), DECIMAL_3)

    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_3)

    def test_predict_xb(self):
        assert_almost_equal(self.res1.predict(which='linear')[:10], self.res2.fittedvalues[:10], DECIMAL_3)

    def test_zstat(self):
        assert_almost_equal(self.res1.tvalues, self.res2.z, DECIMAL_1)

    def test_llf(self):
        assert_almost_equal(self.res1.llf, self.res2.llf, DECIMAL_1)

    def test_llr(self):
        assert_almost_equal(self.res1.llr, self.res2.llr, DECIMAL_2)

    def test_bse(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL_3)