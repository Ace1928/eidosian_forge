import warnings
import sys
import numpy as np
from numpy.random import RandomState
from numpy.testing import (assert_array_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import re
from scipy import optimize
from scipy import stats
from scipy.stats._morestats import _abw_state, _get_As_weibull, _Avals_weibull
from .common_tests import check_named_results
from .._hypotests import _get_wilcoxon_distr, _get_wilcoxon_distr2
from scipy.stats._binomtest import _binary_search_for_binom_tst
from scipy.stats._distr_params import distcont
class TestBinomTestP:
    """
    Tests for stats.binomtest as a replacement for deprecated stats.binom_test.
    """

    @staticmethod
    def binom_test_func(x, n=None, p=0.5, alternative='two-sided'):
        x = np.atleast_1d(x).astype(np.int_)
        if len(x) == 2:
            n = x[1] + x[0]
            x = x[0]
        elif len(x) == 1:
            x = x[0]
            if n is None or n < x:
                raise ValueError('n must be >= x')
            n = np.int_(n)
        else:
            raise ValueError('Incorrect length for x.')
        result = stats.binomtest(x, n, p=p, alternative=alternative)
        return result.pvalue

    def test_data(self):
        pval = self.binom_test_func(100, 250)
        assert_almost_equal(pval, 0.0018833009350757682, 11)
        pval = self.binom_test_func(201, 405)
        assert_almost_equal(pval, 0.9208520596267071, 11)
        pval = self.binom_test_func([682, 243], p=3 / 4)
        assert_almost_equal(pval, 0.38249155957481695, 11)

    def test_bad_len_x(self):
        assert_raises(ValueError, self.binom_test_func, [1, 2, 3])

    def test_bad_n(self):
        assert_raises(ValueError, self.binom_test_func, [100])
        assert_raises(ValueError, self.binom_test_func, [100], n=50)

    def test_bad_p(self):
        assert_raises(ValueError, self.binom_test_func, [50, 50], p=2.0)

    def test_alternatives(self):
        res = self.binom_test_func(51, 235, p=1 / 6, alternative='less')
        assert_almost_equal(res, 0.982022657605858)
        res = self.binom_test_func(51, 235, p=1 / 6, alternative='greater')
        assert_almost_equal(res, 0.02654424571169085)
        res = self.binom_test_func(51, 235, p=1 / 6, alternative='two-sided')
        assert_almost_equal(res, 0.0437479701823997)

    @pytest.mark.skipif(sys.maxsize <= 2 ** 32, reason='32-bit does not overflow')
    def test_boost_overflow_raises(self):
        assert_raises(OverflowError, self.binom_test_func, 5.0, 6, p=sys.float_info.min)