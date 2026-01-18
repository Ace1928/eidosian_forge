import itertools
import os
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_,
from pytest import raises as assert_raises
import pytest
from scipy._lib._testutils import check_free_memory
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate._fitpack_py import (splrep, splev, bisplrep, bisplev,
from scipy.interpolate.dfitpack import regrid_smth
from scipy.interpolate._fitpack2 import dfitpack_int
class TestSmokeTests:
    """
    Smoke tests (with a few asserts) for fitpack routines -- mostly
    check that they are runnable
    """

    def check_1(self, per=0, s=0, a=0, b=2 * np.pi, at_nodes=False, xb=None, xe=None):
        if xb is None:
            xb = a
        if xe is None:
            xe = b
        N = 20
        x = np.linspace(a, b, N + 1)
        x1 = a + (b - a) * np.arange(1, N, dtype=float) / float(N - 1)
        v = f1(x)

        def err_est(k, d):
            h = 1.0 / N
            tol = 5 * h ** (0.75 * (k - d))
            if s > 0:
                tol += 100000.0 * s
            return tol
        for k in range(1, 6):
            tck = splrep(x, v, s=s, per=per, k=k, xe=xe)
            tt = tck[0][k:-k] if at_nodes else x1
            for d in range(k + 1):
                tol = err_est(k, d)
                err = norm2(f1(tt, d) - splev(tt, tck, d)) / norm2(f1(tt, d))
                assert err < tol

    def check_2(self, per=0, N=20, ia=0, ib=2 * np.pi):
        a, b, dx = (0, 2 * np.pi, 0.2 * np.pi)
        x = np.linspace(a, b, N + 1)
        v = np.sin(x)

        def err_est(k, d):
            h = 1.0 / N
            tol = 5 * h ** (0.75 * (k - d))
            return tol
        nk = []
        for k in range(1, 6):
            tck = splrep(x, v, s=0, per=per, k=k, xe=b)
            nk.append([splint(ia, ib, tck), spalde(dx, tck)])
        k = 1
        for r in nk:
            d = 0
            for dr in r[1]:
                tol = err_est(k, d)
                assert_allclose(dr, f1(dx, d), atol=0, rtol=tol)
                d = d + 1
            k = k + 1

    def test_smoke_splrep_splev(self):
        self.check_1(s=1e-06)
        self.check_1(b=1.5 * np.pi)
        self.check_1(b=1.5 * np.pi, xe=2 * np.pi, per=1, s=0.1)

    @pytest.mark.parametrize('per', [0, 1])
    @pytest.mark.parametrize('at_nodes', [True, False])
    def test_smoke_splrep_splev_2(self, per, at_nodes):
        self.check_1(per=per, at_nodes=at_nodes)

    @pytest.mark.parametrize('N', [20, 50])
    @pytest.mark.parametrize('per', [0, 1])
    def test_smoke_splint_spalde(self, N, per):
        self.check_2(per=per, N=N)

    @pytest.mark.parametrize('N', [20, 50])
    @pytest.mark.parametrize('per', [0, 1])
    def test_smoke_splint_spalde_iaib(self, N, per):
        self.check_2(ia=0.2 * np.pi, ib=np.pi, N=N, per=per)

    def test_smoke_sproot(self):
        a, b = (0.1, 15)
        x = np.linspace(a, b, 20)
        v = np.sin(x)
        for k in [1, 2, 4, 5]:
            tck = splrep(x, v, s=0, per=0, k=k, xe=b)
            with assert_raises(ValueError):
                sproot(tck)
        k = 3
        tck = splrep(x, v, s=0, k=3)
        roots = sproot(tck)
        assert_allclose(splev(roots, tck), 0, atol=1e-10, rtol=1e-10)
        assert_allclose(roots, np.pi * np.array([1, 2, 3, 4]), rtol=0.001)

    @pytest.mark.parametrize('N', [20, 50])
    @pytest.mark.parametrize('k', [1, 2, 3, 4, 5])
    def test_smoke_splprep_splrep_splev(self, N, k):
        a, b, dx = (0, 2.0 * np.pi, 0.2 * np.pi)
        x = np.linspace(a, b, N + 1)
        v = np.sin(x)
        tckp, u = splprep([x, v], s=0, per=0, k=k, nest=-1)
        uv = splev(dx, tckp)
        err1 = abs(uv[1] - np.sin(uv[0]))
        assert err1 < 0.01
        tck = splrep(x, v, s=0, per=0, k=k)
        err2 = abs(splev(uv[0], tck) - np.sin(uv[0]))
        assert err2 < 0.01
        if k == 3:
            tckp, u = splprep([x, v], s=0, per=0, k=k, nest=-1)
            for d in range(1, k + 1):
                uv = splev(dx, tckp, d)

    def test_smoke_bisplrep_bisplev(self):
        xb, xe = (0, 2.0 * np.pi)
        yb, ye = (0, 2.0 * np.pi)
        kx, ky = (3, 3)
        Nx, Ny = (20, 20)

        def f2(x, y):
            return np.sin(x + y)
        x = np.linspace(xb, xe, Nx + 1)
        y = np.linspace(yb, ye, Ny + 1)
        xy = makepairs(x, y)
        tck = bisplrep(xy[0], xy[1], f2(xy[0], xy[1]), s=0, kx=kx, ky=ky)
        tt = [tck[0][kx:-kx], tck[1][ky:-ky]]
        t2 = makepairs(tt[0], tt[1])
        v1 = bisplev(tt[0], tt[1], tck)
        v2 = f2(t2[0], t2[1])
        v2.shape = (len(tt[0]), len(tt[1]))
        assert norm2(np.ravel(v1 - v2)) < 0.01