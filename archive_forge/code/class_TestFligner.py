import warnings
import sys
from functools import partial
import numpy as np
from numpy.random import RandomState
from numpy.testing import (assert_array_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import re
from scipy import optimize, stats, special
from scipy.stats._morestats import _abw_state, _get_As_weibull, _Avals_weibull
from .common_tests import check_named_results
from .._hypotests import _get_wilcoxon_distr, _get_wilcoxon_distr2
from scipy.stats._binomtest import _binary_search_for_binom_tst
from scipy.stats._distr_params import distcont
class TestFligner:

    def test_data(self):
        x1 = np.arange(5)
        assert_array_almost_equal(stats.fligner(x1, x1 ** 2), (3.2282229927203536, 0.07237918784820788), 11)

    def test_trimmed1(self):
        rs = np.random.RandomState(123)

        def _perturb(g):
            return (np.asarray(g) + 1e-10 * rs.randn(len(g))).tolist()
        g1_ = _perturb(g1)
        g2_ = _perturb(g2)
        g3_ = _perturb(g3)
        Xsq1, pval1 = stats.fligner(g1_, g2_, g3_, center='mean')
        Xsq2, pval2 = stats.fligner(g1_, g2_, g3_, center='trimmed', proportiontocut=0.0)
        assert_almost_equal(Xsq1, Xsq2)
        assert_almost_equal(pval1, pval2)

    def test_trimmed2(self):
        x = [1.2, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0]
        y = [0.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 200.0]
        Xsq1, pval1 = stats.fligner(x, y, center='trimmed', proportiontocut=0.125)
        Xsq2, pval2 = stats.fligner(x[1:-1], y[1:-1], center='mean')
        assert_almost_equal(Xsq1, Xsq2)
        assert_almost_equal(pval1, pval2)

    def test_bad_keyword(self):
        x = np.linspace(-1, 1, 21)
        assert_raises(TypeError, stats.fligner, x, x, portiontocut=0.1)

    def test_bad_center_value(self):
        x = np.linspace(-1, 1, 21)
        assert_raises(ValueError, stats.fligner, x, x, center='trim')

    def test_bad_num_args(self):
        assert_raises(ValueError, stats.fligner, [1])

    def test_empty_arg(self):
        x = np.arange(5)
        assert_equal((np.nan, np.nan), stats.fligner(x, x ** 2, []))