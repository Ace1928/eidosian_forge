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
class TestProbplot:

    def test_basic(self):
        x = stats.norm.rvs(size=20, random_state=12345)
        osm, osr = stats.probplot(x, fit=False)
        osm_expected = [-1.8241636, -1.38768012, -1.11829229, -0.91222575, -0.73908135, -0.5857176, -0.44506467, -0.31273668, -0.18568928, -0.06158146, 0.06158146, 0.18568928, 0.31273668, 0.44506467, 0.5857176, 0.73908135, 0.91222575, 1.11829229, 1.38768012, 1.8241636]
        assert_allclose(osr, np.sort(x))
        assert_allclose(osm, osm_expected)
        res, res_fit = stats.probplot(x, fit=True)
        res_fit_expected = [1.05361841, 0.31297795, 0.98741609]
        assert_allclose(res_fit, res_fit_expected)

    def test_sparams_keyword(self):
        x = stats.norm.rvs(size=100, random_state=123456)
        osm1, osr1 = stats.probplot(x, sparams=None, fit=False)
        osm2, osr2 = stats.probplot(x, sparams=0, fit=False)
        osm3, osr3 = stats.probplot(x, sparams=(), fit=False)
        assert_allclose(osm1, osm2)
        assert_allclose(osm1, osm3)
        assert_allclose(osr1, osr2)
        assert_allclose(osr1, osr3)
        osm, osr = stats.probplot(x, sparams=(), fit=False)

    def test_dist_keyword(self):
        x = stats.norm.rvs(size=20, random_state=12345)
        osm1, osr1 = stats.probplot(x, fit=False, dist='t', sparams=(3,))
        osm2, osr2 = stats.probplot(x, fit=False, dist=stats.t, sparams=(3,))
        assert_allclose(osm1, osm2)
        assert_allclose(osr1, osr2)
        assert_raises(ValueError, stats.probplot, x, dist='wrong-dist-name')
        assert_raises(AttributeError, stats.probplot, x, dist=[])

        class custom_dist:
            """Some class that looks just enough like a distribution."""

            def ppf(self, q):
                return stats.norm.ppf(q, loc=2)
        osm1, osr1 = stats.probplot(x, sparams=(2,), fit=False)
        osm2, osr2 = stats.probplot(x, dist=custom_dist(), fit=False)
        assert_allclose(osm1, osm2)
        assert_allclose(osr1, osr2)

    @pytest.mark.skipif(not have_matplotlib, reason='no matplotlib')
    def test_plot_kwarg(self):
        fig = plt.figure()
        fig.add_subplot(111)
        x = stats.t.rvs(3, size=100, random_state=7654321)
        res1, fitres1 = stats.probplot(x, plot=plt)
        plt.close()
        res2, fitres2 = stats.probplot(x, plot=None)
        res3 = stats.probplot(x, fit=False, plot=plt)
        plt.close()
        res4 = stats.probplot(x, fit=False, plot=None)
        assert_(len(res1) == len(res2) == len(res3) == len(res4) == 2)
        assert_allclose(res1, res2)
        assert_allclose(res1, res3)
        assert_allclose(res1, res4)
        assert_allclose(fitres1, fitres2)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        stats.probplot(x, fit=False, plot=ax)
        plt.close()

    def test_probplot_bad_args(self):
        assert_raises(ValueError, stats.probplot, [1], dist='plate_of_shrimp')

    def test_empty(self):
        assert_equal(stats.probplot([], fit=False), (np.array([]), np.array([])))
        assert_equal(stats.probplot([], fit=True), ((np.array([]), np.array([])), (np.nan, np.nan, 0.0)))

    def test_array_of_size_one(self):
        with np.errstate(invalid='ignore'):
            assert_equal(stats.probplot([1], fit=True), ((np.array([0.0]), np.array([1])), (np.nan, np.nan, 0.0)))