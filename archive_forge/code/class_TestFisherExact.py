import os
import re
import warnings
from collections import namedtuple
from itertools import product
import hypothesis.extra.numpy as npst
import hypothesis
import contextlib
from numpy.testing import (assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
import numpy.ma.testutils as mat
from numpy import array, arange, float32, float64, power
import numpy as np
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.stats._mstats_basic as mstats_basic
from scipy.stats._ksstats import kolmogn
from scipy.special._testutils import FuncData
from scipy.special import binom
from scipy import optimize
from .common_tests import check_named_results
from scipy.spatial.distance import cdist
from scipy.stats._axis_nan_policy import _broadcast_concatenate
from scipy.stats._stats_py import _permutation_distribution_t
from scipy._lib._util import AxisError
class TestFisherExact:
    """Some tests to show that fisher_exact() works correctly.

    Note that in SciPy 0.9.0 this was not working well for large numbers due to
    inaccuracy of the hypergeom distribution (see #1218). Fixed now.

    Also note that R and SciPy have different argument formats for their
    hypergeometric distribution functions.

    R:
    > phyper(18999, 99000, 110000, 39000, lower.tail = FALSE)
    [1] 1.701815e-09
    """

    def test_basic(self):
        fisher_exact = stats.fisher_exact
        res = fisher_exact([[14500, 20000], [30000, 40000]])[1]
        assert_approx_equal(res, 0.01106, significant=4)
        res = fisher_exact([[100, 2], [1000, 5]])[1]
        assert_approx_equal(res, 0.1301, significant=4)
        res = fisher_exact([[2, 7], [8, 2]])[1]
        assert_approx_equal(res, 0.0230141, significant=6)
        res = fisher_exact([[5, 1], [10, 10]])[1]
        assert_approx_equal(res, 0.1973244, significant=6)
        res = fisher_exact([[5, 15], [20, 20]])[1]
        assert_approx_equal(res, 0.0958044, significant=6)
        res = fisher_exact([[5, 16], [20, 25]])[1]
        assert_approx_equal(res, 0.1725862, significant=6)
        res = fisher_exact([[10, 5], [10, 1]])[1]
        assert_approx_equal(res, 0.1973244, significant=6)
        res = fisher_exact([[5, 0], [1, 4]])[1]
        assert_approx_equal(res, 0.04761904, significant=6)
        res = fisher_exact([[0, 1], [3, 2]])[1]
        assert_approx_equal(res, 1.0)
        res = fisher_exact([[0, 2], [6, 4]])[1]
        assert_approx_equal(res, 0.4545454545)
        res = fisher_exact([[2, 7], [8, 2]])
        assert_approx_equal(res[1], 0.0230141, significant=6)
        assert_approx_equal(res[0], 4.0 / 56)

    def test_precise(self):
        tablist = [([[100, 2], [1000, 5]], (0.2505583993422285, 0.1300759363430016)), ([[2, 7], [8, 2]], (0.08586235135736206, 0.02301413756522114)), ([[5, 1], [10, 10]], (4.725646047336584, 0.197324414715719)), ([[5, 15], [20, 20]], (0.3394396617440852, 0.09580440012477637)), ([[5, 16], [20, 25]], (0.3960558326183334, 0.1725864953812994)), ([[10, 5], [10, 1]], (0.2116112781158483, 0.197324414715719)), ([[10, 5], [10, 0]], (0.0, 0.06126482213438734)), ([[5, 0], [1, 4]], (np.inf, 0.04761904761904762)), ([[0, 5], [1, 4]], (0.0, 1.0)), ([[5, 1], [0, 4]], (np.inf, 0.04761904761904758)), ([[0, 1], [3, 2]], (0.0, 1.0))]
        for table, res_r in tablist:
            res = stats.fisher_exact(np.asarray(table))
            np.testing.assert_almost_equal(res[1], res_r[1], decimal=11, verbose=True)

    def test_gh4130(self):
        x = [[6, 37], [108, 200]]
        res = stats.fisher_exact(x)
        assert_allclose(res[1], 0.005092697748126)
        x = [[22, 0], [0, 102]]
        res = stats.fisher_exact(x)
        assert_allclose(res[1], 7.175066786244549e-25)
        x = [[94, 48], [3577, 16988]]
        res = stats.fisher_exact(x)
        assert_allclose(res[1], 2.069356340993818e-37)

    def test_gh9231(self):
        x = [[5829225, 5692693], [5760959, 5760959]]
        res = stats.fisher_exact(x)
        assert_allclose(res[1], 0, atol=1e-170)

    @pytest.mark.slow
    def test_large_numbers(self):
        pvals = [5.56e-11, 2.666e-11, 1.363e-11]
        for pval, num in zip(pvals, [75, 76, 77]):
            res = stats.fisher_exact([[17704, 496], [1065, num]])[1]
            assert_approx_equal(res, pval, significant=4)
        res = stats.fisher_exact([[18000, 80000], [20000, 90000]])[1]
        assert_approx_equal(res, 0.2751, significant=4)

    def test_raises(self):
        assert_raises(ValueError, stats.fisher_exact, np.arange(6).reshape(2, 3))

    def test_row_or_col_zero(self):
        tables = ([[0, 0], [5, 10]], [[5, 10], [0, 0]], [[0, 5], [0, 10]], [[5, 0], [10, 0]])
        for table in tables:
            oddsratio, pval = stats.fisher_exact(table)
            assert_equal(pval, 1.0)
            assert_equal(oddsratio, np.nan)

    def test_less_greater(self):
        tables = ([[2, 7], [8, 2]], [[200, 7], [8, 300]], [[28, 21], [6, 1957]], [[190, 800], [200, 900]], [[0, 2], [3, 0]], [[1, 1], [2, 1]], [[2, 0], [1, 2]], [[0, 1], [2, 3]], [[1, 0], [1, 4]])
        pvals = ([0.0185217259520665, 0.9990149169715733], [1.0, 2.0056578803889148e-122], [1.0, 5.728437460831983e-44], [0.7416227, 0.2959826], [0.1, 1.0], [0.7, 0.9], [1.0, 0.3], [2.0 / 3, 1.0], [1.0, 1.0 / 3])
        for table, pval in zip(tables, pvals):
            res = []
            res.append(stats.fisher_exact(table, alternative='less')[1])
            res.append(stats.fisher_exact(table, alternative='greater')[1])
            assert_allclose(res, pval, atol=0, rtol=1e-07)

    def test_gh3014(self):
        odds, pvalue = stats.fisher_exact([[1, 2], [9, 84419233]])

    @pytest.mark.parametrize('alternative', ['two-sided', 'less', 'greater'])
    def test_result(self, alternative):
        table = np.array([[14500, 20000], [30000, 40000]])
        res = stats.fisher_exact(table, alternative=alternative)
        assert_equal((res.statistic, res.pvalue), res)