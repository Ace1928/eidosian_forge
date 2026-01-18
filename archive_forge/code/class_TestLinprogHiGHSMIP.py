import sys
import platform
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import MatrixRankWarning
from scipy.linalg import LinAlgWarning
from scipy._lib._util import VisibleDeprecationWarning
import scipy.sparse
import pytest
class TestLinprogHiGHSMIP:
    method = 'highs'
    options = {}

    @pytest.mark.xfail(condition=sys.maxsize < 2 ** 32 and platform.system() == 'Linux', run=False, reason='gh-16347')
    def test_mip1(self):
        n = 4
        A, b, c, numbers, M = magic_square(n)
        bounds = [(0, 1)] * len(c)
        integrality = [1] * len(c)
        res = linprog(c=c * 0, A_eq=A, b_eq=b, bounds=bounds, method=self.method, integrality=integrality)
        s = (numbers.flatten() * res.x).reshape(n ** 2, n, n)
        square = np.sum(s, axis=0)
        np.testing.assert_allclose(square.sum(axis=0), M)
        np.testing.assert_allclose(square.sum(axis=1), M)
        np.testing.assert_allclose(np.diag(square).sum(), M)
        np.testing.assert_allclose(np.diag(square[:, ::-1]).sum(), M)
        np.testing.assert_allclose(res.x, np.round(res.x), atol=1e-12)

    def test_mip2(self):
        A_ub = np.array([[2, -2], [-8, 10]])
        b_ub = np.array([-1, 13])
        c = -np.array([1, 1])
        bounds = np.array([(0, np.inf)] * len(c))
        integrality = np.ones_like(c)
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=self.method, integrality=integrality)
        np.testing.assert_allclose(res.x, [1, 2])
        np.testing.assert_allclose(res.fun, -3)

    def test_mip3(self):
        A_ub = np.array([[-1, 1], [3, 2], [2, 3]])
        b_ub = np.array([1, 12, 12])
        c = -np.array([0, 1])
        bounds = [(0, np.inf)] * len(c)
        integrality = [1] * len(c)
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=self.method, integrality=integrality)
        np.testing.assert_allclose(res.fun, -2)
        assert np.allclose(res.x, [1, 2]) or np.allclose(res.x, [2, 2])

    def test_mip4(self):
        A_ub = np.array([[-1, -2], [-4, -1], [2, 1]])
        b_ub = np.array([14, -33, 20])
        c = np.array([8, 1])
        bounds = [(0, np.inf)] * len(c)
        integrality = [0, 1]
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=self.method, integrality=integrality)
        np.testing.assert_allclose(res.x, [6.5, 7])
        np.testing.assert_allclose(res.fun, 59)

    def test_mip5(self):
        A_ub = np.array([[1, 1, 1]])
        b_ub = np.array([7])
        A_eq = np.array([[4, 2, 1]])
        b_eq = np.array([12])
        c = np.array([-3, -2, -1])
        bounds = [(0, np.inf), (0, np.inf), (0, 1)]
        integrality = [0, 1, 0]
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, integrality=integrality)
        np.testing.assert_allclose(res.x, [0, 6, 0])
        np.testing.assert_allclose(res.fun, -12)
        assert res.get('mip_node_count', None) is not None
        assert res.get('mip_dual_bound', None) is not None
        assert res.get('mip_gap', None) is not None

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_mip6(self):
        A_eq = np.array([[22, 13, 26, 33, 21, 3, 14, 26], [39, 16, 22, 28, 26, 30, 23, 24], [18, 14, 29, 27, 30, 38, 26, 26], [41, 26, 28, 36, 18, 38, 16, 26]])
        b_eq = np.array([7872, 10466, 11322, 12058])
        c = np.array([2, 10, 13, 17, 7, 5, 7, 3])
        bounds = [(0, np.inf)] * 8
        integrality = [1] * 8
        res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, integrality=integrality)
        np.testing.assert_allclose(res.fun, 1854)

    @pytest.mark.xslow
    def test_mip_rel_gap_passdown(self):
        A_eq = np.array([[22, 13, 26, 33, 21, 3, 14, 26], [39, 16, 22, 28, 26, 30, 23, 24], [18, 14, 29, 27, 30, 38, 26, 26], [41, 26, 28, 36, 18, 38, 16, 26]])
        b_eq = np.array([7872, 10466, 11322, 12058])
        c = np.array([2, 10, 13, 17, 7, 5, 7, 3])
        bounds = [(0, np.inf)] * 8
        integrality = [1] * 8
        mip_rel_gaps = [0.5, 0.25, 0.01, 0.001]
        sol_mip_gaps = []
        for mip_rel_gap in mip_rel_gaps:
            res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, integrality=integrality, options={'mip_rel_gap': mip_rel_gap})
            final_mip_gap = res['mip_gap']
            assert final_mip_gap <= mip_rel_gap
            sol_mip_gaps.append(final_mip_gap)
        gap_diffs = np.diff(np.flip(sol_mip_gaps))
        assert np.all(gap_diffs >= 0)
        assert not np.all(gap_diffs == 0)

    def test_semi_continuous(self):
        c = np.array([1.0, 1.0, -1, -1])
        bounds = np.array([[0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [0.5, 1.5]])
        integrality = np.array([2, 3, 2, 3])
        res = linprog(c, bounds=bounds, integrality=integrality, method='highs')
        np.testing.assert_allclose(res.x, [0, 0, 1.5, 1])
        assert res.status == 0