from itertools import product
import numpy as np
import random
import functools
import pytest
from numpy.testing import (assert_, assert_equal, assert_allclose,
from pytest import raises as assert_raises
import scipy.stats as stats
from scipy.stats import distributions
from scipy.stats._hypotests import (epps_singleton_2samp, cramervonmises,
from scipy.stats._mannwhitneyu import mannwhitneyu, _mwu_state
from .common_tests import check_named_results
from scipy._lib._testutils import _TestPythranFunc
class TestSomersD(_TestPythranFunc):

    def setup_method(self):
        self.dtypes = self.ALL_INTEGER + self.ALL_FLOAT
        self.arguments = {0: (np.arange(10), self.ALL_INTEGER + self.ALL_FLOAT), 1: (np.arange(10), self.ALL_INTEGER + self.ALL_FLOAT)}
        input_array = [self.arguments[idx][0] for idx in self.arguments]
        self.partialfunc = functools.partial(stats.somersd, alternative='two-sided')
        self.expected = self.partialfunc(*input_array)

    def pythranfunc(self, *args):
        res = self.partialfunc(*args)
        assert_allclose(res.statistic, self.expected.statistic, atol=1e-15)
        assert_allclose(res.pvalue, self.expected.pvalue, atol=1e-15)

    def test_pythranfunc_keywords(self):
        table = [[27, 25, 14, 7, 0], [7, 14, 18, 35, 12], [1, 3, 2, 7, 17]]
        res1 = stats.somersd(table)
        optional_args = self.get_optional_args(stats.somersd)
        res2 = stats.somersd(table, **optional_args)
        assert_allclose(res1.statistic, res2.statistic, atol=1e-15)
        assert_allclose(res1.pvalue, res2.pvalue, atol=1e-15)

    def test_like_kendalltau(self):
        x = [5, 2, 1, 3, 6, 4, 7, 8]
        y = [5, 2, 6, 3, 1, 8, 7, 4]
        expected = (0.0, 1.0)
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)
        x = [0, 5, 2, 1, 3, 6, 4, 7, 8]
        y = [5, 2, 0, 6, 3, 1, 8, 7, 4]
        expected = (0.0, 1.0)
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)
        x = [5, 2, 1, 3, 6, 4, 7]
        y = [5, 2, 6, 3, 1, 7, 4]
        expected = (-0.14285714285714, 0.63032695315767)
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)
        x = np.arange(10)
        y = np.arange(10)
        expected = (1.0, 0)
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)
        x = np.arange(10)
        y = np.array([0, 2, 1, 3, 4, 6, 5, 7, 8, 9])
        expected = (0.91111111111111, 0.0)
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)
        x = np.arange(10)
        y = np.arange(10)[::-1]
        expected = (-1.0, 0)
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)
        x = np.arange(10)
        y = np.array([9, 7, 8, 6, 5, 3, 4, 2, 1, 0])
        expected = (-0.9111111111111111, 0.0)
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)
        x1 = [12, 2, 1, 12, 2]
        x2 = [1, 4, 7, 1, 0]
        expected = (-0.5, 0.30490178817878)
        res = stats.somersd(x1, x2)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)
        res = stats.somersd([2, 2, 2], [2, 2, 2])
        assert_allclose(res.statistic, np.nan)
        assert_allclose(res.pvalue, np.nan)
        res = stats.somersd([2, 0, 2], [2, 2, 2])
        assert_allclose(res.statistic, np.nan)
        assert_allclose(res.pvalue, np.nan)
        res = stats.somersd([2, 2, 2], [2, 0, 2])
        assert_allclose(res.statistic, np.nan)
        assert_allclose(res.pvalue, np.nan)
        res = stats.somersd([0], [0])
        assert_allclose(res.statistic, np.nan)
        assert_allclose(res.pvalue, np.nan)
        res = stats.somersd([], [])
        assert_allclose(res.statistic, np.nan)
        assert_allclose(res.pvalue, np.nan)
        x = np.arange(10.0)
        y = np.arange(20.0)
        assert_raises(ValueError, stats.somersd, x, y)

    def test_asymmetry(self):
        x = [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
        y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        d_cr = 0.27272727272727
        d_rc = 0.34285714285714
        p = 0.0928919408837
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, d_cr, atol=1e-15)
        assert_allclose(res.pvalue, p, atol=0.0001)
        assert_equal(res.table.shape, (3, 2))
        res = stats.somersd(y, x)
        assert_allclose(res.statistic, d_rc, atol=1e-15)
        assert_allclose(res.pvalue, p, atol=1e-15)
        assert_equal(res.table.shape, (2, 3))

    def test_somers_original(self):
        table = np.array([[8, 2], [6, 5], [3, 4], [1, 3], [2, 3]])
        table = table.T
        dyx = 129 / 340
        assert_allclose(stats.somersd(table).statistic, dyx)
        table = np.array([[25, 0], [85, 0], [0, 30]])
        dxy, dyx = (3300 / 5425, 3300 / 3300)
        assert_allclose(stats.somersd(table).statistic, dxy)
        assert_allclose(stats.somersd(table.T).statistic, dyx)
        table = np.array([[25, 0], [0, 30], [85, 0]])
        dyx = -1800 / 3300
        assert_allclose(stats.somersd(table.T).statistic, dyx)

    def test_contingency_table_with_zero_rows_cols(self):
        N = 100
        shape = (4, 6)
        size = np.prod(shape)
        np.random.seed(0)
        s = stats.multinomial.rvs(N, p=np.ones(size) / size).reshape(shape)
        res = stats.somersd(s)
        s2 = np.insert(s, 2, np.zeros(shape[1]), axis=0)
        res2 = stats.somersd(s2)
        s3 = np.insert(s, 2, np.zeros(shape[0]), axis=1)
        res3 = stats.somersd(s3)
        s4 = np.insert(s2, 2, np.zeros(shape[0] + 1), axis=1)
        res4 = stats.somersd(s4)
        assert_allclose(res.statistic, -0.11698113207547, atol=1e-15)
        assert_allclose(res.statistic, res2.statistic)
        assert_allclose(res.statistic, res3.statistic)
        assert_allclose(res.statistic, res4.statistic)
        assert_allclose(res.pvalue, 0.15637644818815, atol=1e-15)
        assert_allclose(res.pvalue, res2.pvalue)
        assert_allclose(res.pvalue, res3.pvalue)
        assert_allclose(res.pvalue, res4.pvalue)

    def test_invalid_contingency_tables(self):
        N = 100
        shape = (4, 6)
        size = np.prod(shape)
        np.random.seed(0)
        s = stats.multinomial.rvs(N, p=np.ones(size) / size).reshape(shape)
        s5 = s - 2
        message = 'All elements of the contingency table must be non-negative'
        with assert_raises(ValueError, match=message):
            stats.somersd(s5)
        s6 = s + 0.01
        message = 'All elements of the contingency table must be integer'
        with assert_raises(ValueError, match=message):
            stats.somersd(s6)
        message = 'At least two elements of the contingency table must be nonzero.'
        with assert_raises(ValueError, match=message):
            stats.somersd([[]])
        with assert_raises(ValueError, match=message):
            stats.somersd([[1]])
        s7 = np.zeros((3, 3))
        with assert_raises(ValueError, match=message):
            stats.somersd(s7)
        s7[0, 1] = 1
        with assert_raises(ValueError, match=message):
            stats.somersd(s7)

    def test_only_ranks_matter(self):
        x = [1, 2, 3]
        x2 = [-1, 2.1, np.inf]
        y = [3, 2, 1]
        y2 = [0, -0.5, -np.inf]
        res = stats.somersd(x, y)
        res2 = stats.somersd(x2, y2)
        assert_equal(res.statistic, res2.statistic)
        assert_equal(res.pvalue, res2.pvalue)

    def test_contingency_table_return(self):
        x = np.arange(10)
        y = np.arange(10)
        res = stats.somersd(x, y)
        assert_equal(res.table, np.eye(10))

    def test_somersd_alternative(self):
        x1 = [1, 2, 3, 4, 5]
        x2 = [5, 6, 7, 8, 7]
        expected = stats.somersd(x1, x2, alternative='two-sided')
        assert expected.statistic > 0
        res = stats.somersd(x1, x2, alternative='less')
        assert_equal(res.statistic, expected.statistic)
        assert_allclose(res.pvalue, 1 - expected.pvalue / 2)
        res = stats.somersd(x1, x2, alternative='greater')
        assert_equal(res.statistic, expected.statistic)
        assert_allclose(res.pvalue, expected.pvalue / 2)
        x2.reverse()
        expected = stats.somersd(x1, x2, alternative='two-sided')
        assert expected.statistic < 0
        res = stats.somersd(x1, x2, alternative='greater')
        assert_equal(res.statistic, expected.statistic)
        assert_allclose(res.pvalue, 1 - expected.pvalue / 2)
        res = stats.somersd(x1, x2, alternative='less')
        assert_equal(res.statistic, expected.statistic)
        assert_allclose(res.pvalue, expected.pvalue / 2)
        with pytest.raises(ValueError, match="alternative must be 'less'..."):
            stats.somersd(x1, x2, alternative='ekki-ekki')

    @pytest.mark.parametrize('positive_correlation', (False, True))
    def test_somersd_perfect_correlation(self, positive_correlation):
        x1 = np.arange(10)
        x2 = x1 if positive_correlation else np.flip(x1)
        expected_statistic = 1 if positive_correlation else -1
        res = stats.somersd(x1, x2, alternative='two-sided')
        assert res.statistic == expected_statistic
        assert res.pvalue == 0
        res = stats.somersd(x1, x2, alternative='less')
        assert res.statistic == expected_statistic
        assert res.pvalue == (1 if positive_correlation else 0)
        res = stats.somersd(x1, x2, alternative='greater')
        assert res.statistic == expected_statistic
        assert res.pvalue == (0 if positive_correlation else 1)

    def test_somersd_large_inputs_gh18132(self):
        classes = [1, 2]
        n_samples = 10 ** 6
        random.seed(6272161)
        x = random.choices(classes, k=n_samples)
        y = random.choices(classes, k=n_samples)
        val_sklearn = -0.001528138777036947
        val_scipy = stats.somersd(x, y).statistic
        assert_allclose(val_sklearn, val_scipy, atol=1e-15)