import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.stats.proportion as smprop
from statsmodels.stats.proportion import (
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.stats.tests.results.results_proportion import res_binom, res_binom_methods
class CheckProportionMixin:

    def test_proptest(self):
        pt = smprop.proportions_chisquare(self.n_success, self.nobs, value=None)
        assert_almost_equal(pt[0], self.res_prop_test.statistic, decimal=13)
        assert_almost_equal(pt[1], self.res_prop_test.p_value, decimal=13)
        pt = smprop.proportions_chisquare(self.n_success, self.nobs, value=self.res_prop_test_val.null_value[0])
        assert_almost_equal(pt[0], self.res_prop_test_val.statistic, decimal=13)
        assert_almost_equal(pt[1], self.res_prop_test_val.p_value, decimal=13)
        pt = smprop.proportions_chisquare(self.n_success[0], self.nobs[0], value=self.res_prop_test_1.null_value)
        assert_almost_equal(pt[0], self.res_prop_test_1.statistic, decimal=13)
        assert_almost_equal(pt[1], self.res_prop_test_1.p_value, decimal=13)

    def test_pairwiseproptest(self):
        ppt = smprop.proportions_chisquare_allpairs(self.n_success, self.nobs, multitest_method=None)
        assert_almost_equal(ppt.pvals_raw, self.res_ppt_pvals_raw)
        ppt = smprop.proportions_chisquare_allpairs(self.n_success, self.nobs, multitest_method='h')
        assert_almost_equal(ppt.pval_corrected(), self.res_ppt_pvals_holm)
        pptd = smprop.proportions_chisquare_pairscontrol(self.n_success, self.nobs, multitest_method='hommel')
        assert_almost_equal(pptd.pvals_raw, ppt.pvals_raw[:len(self.nobs) - 1], decimal=13)

    def test_number_pairs_1493(self):
        ppt = smprop.proportions_chisquare_allpairs(self.n_success[:3], self.nobs[:3], multitest_method=None)
        assert_equal(len(ppt.pvals_raw), 3)
        idx = [0, 1, 3]
        assert_almost_equal(ppt.pvals_raw, self.res_ppt_pvals_raw[idx])