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
class TestNegativeBinomialNB1Newton(CheckNegBinMixin, CheckModelResults):

    @classmethod
    def setup_class(cls):
        data = load_randhie()
        exog = sm.add_constant(data.exog, prepend=False)
        model = NegativeBinomial(data.endog, exog, 'nb1')
        cls.res1 = model.fit(method='newton', maxiter=100, disp=0)
        res2 = RandHIE.negativebinomial_nb1_bfgs
        cls.res2 = res2

    def test_zstat(self):
        assert_almost_equal(self.res1.tvalues, self.res2.z, DECIMAL_1)

    def test_lnalpha(self):
        self.res1.bse
        assert_almost_equal(self.res1.lnalpha, self.res2.lnalpha, 3)
        assert_almost_equal(self.res1.lnalpha_std_err, self.res2.lnalpha_std_err, DECIMAL_4)

    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)

    def test_conf_int(self):
        assert_almost_equal(self.res1.conf_int(), self.res2.conf_int, DECIMAL_2)

    @pytest.mark.xfail(reason='Test has not been implemented for this class.', strict=True, raises=NotImplementedError)
    def test_predict(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test has not been implemented for this class.', strict=True, raises=NotImplementedError)
    def test_predict_xb(self):
        raise NotImplementedError