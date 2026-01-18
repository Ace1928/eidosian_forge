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
class TestPoissonNewton(CheckModelResults):

    @classmethod
    def setup_class(cls):
        data = load_randhie()
        exog = sm.add_constant(data.exog, prepend=False)
        cls.res1 = Poisson(data.endog, exog).fit(method='newton', disp=0)
        res2 = RandHIE.poisson
        cls.res2 = res2

    def test_margeff_overall(self):
        me = self.res1.get_margeff()
        assert_almost_equal(me.margeff, self.res2.margeff_nodummy_overall, DECIMAL_4)
        assert_almost_equal(me.margeff_se, self.res2.margeff_nodummy_overall_se, DECIMAL_4)

    def test_margeff_dummy_overall(self):
        me = self.res1.get_margeff(dummy=True)
        assert_almost_equal(me.margeff, self.res2.margeff_dummy_overall, DECIMAL_4)
        assert_almost_equal(me.margeff_se, self.res2.margeff_dummy_overall_se, DECIMAL_4)

    def test_resid(self):
        assert_almost_equal(self.res1.resid, self.res2.resid, 2)

    def test_predict_prob(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(cur_dir, 'results', 'predict_prob_poisson.csv')
        probs_res = np.loadtxt(path, delimiter=',')
        probs = self.res1.predict_prob()[:100]
        assert_almost_equal(probs, probs_res, 8)

    @pytest.mark.xfail(reason='res2.cov_params is a zero-dim array of None', strict=True)
    def test_cov_params(self):
        super().test_cov_params()