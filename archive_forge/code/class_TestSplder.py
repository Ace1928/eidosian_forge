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
class TestSplder:

    def setup_method(self):
        x = np.linspace(0, 1, 100) ** 3
        y = np.sin(20 * x)
        self.spl = splrep(x, y)
        assert_(np.ptp(np.diff(self.spl[0])) > 0)

    def test_inverse(self):
        for n in range(5):
            spl2 = splantider(self.spl, n)
            spl3 = splder(spl2, n)
            assert_allclose(self.spl[0], spl3[0])
            assert_allclose(self.spl[1], spl3[1])
            assert_equal(self.spl[2], spl3[2])

    def test_splder_vs_splev(self):
        for n in range(3 + 1):
            xx = np.linspace(-1, 2, 2000)
            if n == 3:
                xx = xx[(xx >= 0) & (xx <= 1)]
            dy = splev(xx, self.spl, n)
            spl2 = splder(self.spl, n)
            dy2 = splev(xx, spl2)
            if n == 1:
                assert_allclose(dy, dy2, rtol=2e-06)
            else:
                assert_allclose(dy, dy2)

    def test_splantider_vs_splint(self):
        spl2 = splantider(self.spl)
        xx = np.linspace(0, 1, 20)
        for x1 in xx:
            for x2 in xx:
                y1 = splint(x1, x2, self.spl)
                y2 = splev(x2, spl2) - splev(x1, spl2)
                assert_allclose(y1, y2)

    def test_order0_diff(self):
        assert_raises(ValueError, splder, self.spl, 4)

    def test_kink(self):
        spl2 = insert(0.5, self.spl, m=2)
        splder(spl2, 2)
        assert_raises(ValueError, splder, spl2, 3)
        spl2 = insert(0.5, self.spl, m=3)
        splder(spl2, 1)
        assert_raises(ValueError, splder, spl2, 2)
        spl2 = insert(0.5, self.spl, m=4)
        assert_raises(ValueError, splder, spl2, 1)

    def test_multidim(self):
        for n in range(3):
            t, c, k = self.spl
            c2 = np.c_[c, c, c]
            c2 = np.dstack((c2, c2))
            spl2 = splantider((t, c2, k), n)
            spl3 = splder(spl2, n)
            assert_allclose(t, spl3[0])
            assert_allclose(c2, spl3[1])
            assert_equal(k, spl3[2])