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
class TestSplint:

    def test_len_c(self):
        n, k = (7, 3)
        x = np.arange(n)
        y = x ** 3
        t, c, k = splrep(x, y, s=0)
        assert len(t) == len(c) == n + 2 * (k - 1)
        res = splint(0, 6, (t, c, k))
        assert_allclose(res, 6 ** 4 / 4, atol=1e-15)
        c0 = c.copy()
        c0[len(t) - k - 1:] = np.nan
        res0 = splint(0, 6, (t, c0, k))
        assert_allclose(res0, 6 ** 4 / 4, atol=1e-15)
        c0[6] = np.nan
        assert np.isnan(splint(0, 6, (t, c0, k)))
        c1 = c[:len(t) - k - 1]
        res1 = splint(0, 6, (t, c1, k))
        assert_allclose(res1, 6 ** 4 / 4, atol=1e-15)
        with assert_raises(Exception, match='>=n-k-1'):
            splint(0, 1, (np.ones(10), np.ones(5), 3))