import pickle
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from .test_continuous_basic import check_distribution_rvs
import numpy
import numpy as np
import scipy.linalg
from scipy.stats._multivariate import (_PSD,
from scipy.stats import (multivariate_normal, multivariate_hypergeom,
from scipy.stats import _covariance, Covariance
from scipy import stats
from scipy.integrate import romb, qmc_quad, tplquad
from scipy.special import multigammaln
from scipy._lib._pep440 import Version
from .common_tests import check_random_state_property
from .data._mvt import _qsimvtv
from unittest.mock import patch
class TestRandomTable:

    def get_rng(self):
        return np.random.default_rng(628174795866951638)

    def test_process_parameters(self):
        message = '`row` must be one-dimensional'
        with pytest.raises(ValueError, match=message):
            random_table([[1, 2]], [1, 2])
        message = '`col` must be one-dimensional'
        with pytest.raises(ValueError, match=message):
            random_table([1, 2], [[1, 2]])
        message = 'each element of `row` must be non-negative'
        with pytest.raises(ValueError, match=message):
            random_table([1, -1], [1, 2])
        message = 'each element of `col` must be non-negative'
        with pytest.raises(ValueError, match=message):
            random_table([1, 2], [1, -2])
        message = 'sums over `row` and `col` must be equal'
        with pytest.raises(ValueError, match=message):
            random_table([1, 2], [1, 0])
        message = 'each element of `row` must be an integer'
        with pytest.raises(ValueError, match=message):
            random_table([2.1, 2.1], [1, 1, 2])
        message = 'each element of `col` must be an integer'
        with pytest.raises(ValueError, match=message):
            random_table([1, 2], [1.1, 1.1, 1])
        row = [1, 3]
        col = [2, 1, 1]
        r, c, n = random_table._process_parameters([1, 3], [2, 1, 1])
        assert_equal(row, r)
        assert_equal(col, c)
        assert n == np.sum(row)

    @pytest.mark.parametrize('scale,method', ((1, 'boyett'), (100, 'patefield')))
    def test_process_rvs_method_on_None(self, scale, method):
        row = np.array([1, 3]) * scale
        col = np.array([2, 1, 1]) * scale
        ct = random_table
        expected = ct.rvs(row, col, method=method, random_state=1)
        got = ct.rvs(row, col, method=None, random_state=1)
        assert_equal(expected, got)

    def test_process_rvs_method_bad_argument(self):
        row = [1, 3]
        col = [2, 1, 1]
        message = "'foo' not recognized, must be one of"
        with pytest.raises(ValueError, match=message):
            random_table.rvs(row, col, method='foo')

    @pytest.mark.parametrize('frozen', (True, False))
    @pytest.mark.parametrize('log', (True, False))
    def test_pmf_logpmf(self, frozen, log):
        rng = self.get_rng()
        row = [2, 6]
        col = [1, 3, 4]
        rvs = random_table.rvs(row, col, size=1000, method='boyett', random_state=rng)
        obj = random_table(row, col) if frozen else random_table
        method = getattr(obj, 'logpmf' if log else 'pmf')
        if not frozen:
            original_method = method

            def method(x):
                return original_method(x, row, col)
        pmf = (lambda x: np.exp(method(x))) if log else method
        unique_rvs, counts = np.unique(rvs, axis=0, return_counts=True)
        p = pmf(unique_rvs)
        assert_allclose(p * len(rvs), counts, rtol=0.1)
        p2 = pmf(list(unique_rvs[0]))
        assert_equal(p2, p[0])
        rvs_nd = rvs.reshape((10, 100) + rvs.shape[1:])
        p = pmf(rvs_nd)
        assert p.shape == (10, 100)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                pij = p[i, j]
                rvij = rvs_nd[i, j]
                qij = pmf(rvij)
                assert_equal(pij, qij)
        x = [[0, 1, 1], [2, 1, 3]]
        assert_equal(np.sum(x, axis=-1), row)
        p = pmf(x)
        assert p == 0
        x = [[0, 1, 2], [1, 2, 2]]
        assert_equal(np.sum(x, axis=-2), col)
        p = pmf(x)
        assert p == 0
        message = '`x` must be at least two-dimensional'
        with pytest.raises(ValueError, match=message):
            pmf([1])
        message = '`x` must contain only integral values'
        with pytest.raises(ValueError, match=message):
            pmf([[1.1]])
        message = '`x` must contain only integral values'
        with pytest.raises(ValueError, match=message):
            pmf([[np.nan]])
        message = '`x` must contain only non-negative values'
        with pytest.raises(ValueError, match=message):
            pmf([[-1]])
        message = 'shape of `x` must agree with `row`'
        with pytest.raises(ValueError, match=message):
            pmf([[1, 2, 3]])
        message = 'shape of `x` must agree with `col`'
        with pytest.raises(ValueError, match=message):
            pmf([[1, 2], [3, 4]])

    @pytest.mark.parametrize('method', ('boyett', 'patefield'))
    def test_rvs_mean(self, method):
        rng = self.get_rng()
        row = [2, 6]
        col = [1, 3, 4]
        rvs = random_table.rvs(row, col, size=1000, method=method, random_state=rng)
        mean = random_table.mean(row, col)
        assert_equal(np.sum(mean), np.sum(row))
        assert_allclose(rvs.mean(0), mean, atol=0.05)
        assert_equal(rvs.sum(axis=-1), np.broadcast_to(row, (1000, 2)))
        assert_equal(rvs.sum(axis=-2), np.broadcast_to(col, (1000, 3)))

    def test_rvs_cov(self):
        rng = self.get_rng()
        row = [2, 6]
        col = [1, 3, 4]
        rvs1 = random_table.rvs(row, col, size=10000, method='boyett', random_state=rng)
        rvs2 = random_table.rvs(row, col, size=10000, method='patefield', random_state=rng)
        cov1 = np.var(rvs1, axis=0)
        cov2 = np.var(rvs2, axis=0)
        assert_allclose(cov1, cov2, atol=0.02)

    @pytest.mark.parametrize('method', ('boyett', 'patefield'))
    def test_rvs_size(self, method):
        row = [2, 6]
        col = [1, 3, 4]
        rv = random_table.rvs(row, col, method=method, random_state=self.get_rng())
        assert rv.shape == (2, 3)
        rv2 = random_table.rvs(row, col, size=1, method=method, random_state=self.get_rng())
        assert rv2.shape == (1, 2, 3)
        assert_equal(rv, rv2[0])
        rv3 = random_table.rvs(row, col, size=0, method=method, random_state=self.get_rng())
        assert rv3.shape == (0, 2, 3)
        rv4 = random_table.rvs(row, col, size=20, method=method, random_state=self.get_rng())
        assert rv4.shape == (20, 2, 3)
        rv5 = random_table.rvs(row, col, size=(4, 5), method=method, random_state=self.get_rng())
        assert rv5.shape == (4, 5, 2, 3)
        assert_allclose(rv5.reshape(20, 2, 3), rv4, rtol=1e-15)
        message = '`size` must be a non-negative integer or `None`'
        with pytest.raises(ValueError, match=message):
            random_table.rvs(row, col, size=-1, method=method, random_state=self.get_rng())
        with pytest.raises(ValueError, match=message):
            random_table.rvs(row, col, size=np.nan, method=method, random_state=self.get_rng())

    @pytest.mark.parametrize('method', ('boyett', 'patefield'))
    def test_rvs_method(self, method):
        row = [2, 6]
        col = [1, 3, 4]
        ct = random_table
        rvs = ct.rvs(row, col, size=100000, method=method, random_state=self.get_rng())
        unique_rvs, counts = np.unique(rvs, axis=0, return_counts=True)
        p = ct.pmf(unique_rvs, row, col)
        assert_allclose(p * len(rvs), counts, rtol=0.02)

    @pytest.mark.parametrize('method', ('boyett', 'patefield'))
    def test_rvs_with_zeros_in_col_row(self, method):
        row = [0, 1, 0]
        col = [1, 0, 0, 0]
        d = random_table(row, col)
        rv = d.rvs(1000, method=method, random_state=self.get_rng())
        expected = np.zeros((1000, len(row), len(col)))
        expected[...] = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
        assert_equal(rv, expected)

    @pytest.mark.parametrize('method', (None, 'boyett', 'patefield'))
    @pytest.mark.parametrize('col', ([], [0]))
    @pytest.mark.parametrize('row', ([], [0]))
    def test_rvs_with_edge_cases(self, method, row, col):
        d = random_table(row, col)
        rv = d.rvs(10, method=method, random_state=self.get_rng())
        expected = np.zeros((10, len(row), len(col)))
        assert_equal(rv, expected)

    @pytest.mark.parametrize('v', (1, 2))
    def test_rvs_rcont(self, v):
        import scipy.stats._rcont as _rcont
        row = np.array([1, 3], dtype=np.int64)
        col = np.array([2, 1, 1], dtype=np.int64)
        rvs = getattr(_rcont, f'rvs_rcont{v}')
        ntot = np.sum(row)
        result = rvs(row, col, ntot, 1, self.get_rng())
        assert result.shape == (1, len(row), len(col))
        assert np.sum(result) == ntot

    def test_frozen(self):
        row = [2, 6]
        col = [1, 3, 4]
        d = random_table(row, col, seed=self.get_rng())
        sample = d.rvs()
        expected = random_table.mean(row, col)
        assert_equal(expected, d.mean())
        expected = random_table.pmf(sample, row, col)
        assert_equal(expected, d.pmf(sample))
        expected = random_table.logpmf(sample, row, col)
        assert_equal(expected, d.logpmf(sample))

    @pytest.mark.parametrize('method', ('boyett', 'patefield'))
    def test_rvs_frozen(self, method):
        row = [2, 6]
        col = [1, 3, 4]
        d = random_table(row, col, seed=self.get_rng())
        expected = random_table.rvs(row, col, size=10, method=method, random_state=self.get_rng())
        got = d.rvs(size=10, method=method)
        assert_equal(expected, got)