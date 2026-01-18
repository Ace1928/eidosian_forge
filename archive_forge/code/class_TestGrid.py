import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
class TestGrid:

    def test_basic(self):
        a = mgrid[-1:1:10j]
        b = mgrid[-1:1:0.1]
        assert_(a.shape == (10,))
        assert_(b.shape == (20,))
        assert_(a[0] == -1)
        assert_almost_equal(a[-1], 1)
        assert_(b[0] == -1)
        assert_almost_equal(b[1] - b[0], 0.1, 11)
        assert_almost_equal(b[-1], b[0] + 19 * 0.1, 11)
        assert_almost_equal(a[1] - a[0], 2.0 / 9.0, 11)

    def test_linspace_equivalence(self):
        y, st = np.linspace(2, 10, retstep=True)
        assert_almost_equal(st, 8 / 49.0)
        assert_array_almost_equal(y, mgrid[2:10:50j], 13)

    def test_nd(self):
        c = mgrid[-1:1:10j, -2:2:10j]
        d = mgrid[-1:1:0.1, -2:2:0.2]
        assert_(c.shape == (2, 10, 10))
        assert_(d.shape == (2, 20, 20))
        assert_array_equal(c[0][0, :], -np.ones(10, 'd'))
        assert_array_equal(c[1][:, 0], -2 * np.ones(10, 'd'))
        assert_array_almost_equal(c[0][-1, :], np.ones(10, 'd'), 11)
        assert_array_almost_equal(c[1][:, -1], 2 * np.ones(10, 'd'), 11)
        assert_array_almost_equal(d[0, 1, :] - d[0, 0, :], 0.1 * np.ones(20, 'd'), 11)
        assert_array_almost_equal(d[1, :, 1] - d[1, :, 0], 0.2 * np.ones(20, 'd'), 11)

    def test_sparse(self):
        grid_full = mgrid[-1:1:10j, -2:2:10j]
        grid_sparse = ogrid[-1:1:10j, -2:2:10j]
        grid_broadcast = np.broadcast_arrays(*grid_sparse)
        for f, b in zip(grid_full, grid_broadcast):
            assert_equal(f, b)

    @pytest.mark.parametrize('start, stop, step, expected', [(None, 10, 10j, (200, 10)), (-10, 20, None, (1800, 30))])
    def test_mgrid_size_none_handling(self, start, stop, step, expected):
        grid = mgrid[start:stop:step, start:stop:step]
        grid_small = mgrid[start:stop:step]
        assert_equal(grid.size, expected[0])
        assert_equal(grid_small.size, expected[1])

    def test_accepts_npfloating(self):
        grid64 = mgrid[0.1:0.33:0.1,]
        grid32 = mgrid[np.float32(0.1):np.float32(0.33):np.float32(0.1),]
        assert_(grid32.dtype == np.float64)
        assert_array_almost_equal(grid64, grid32)
        grid64 = mgrid[0.1:0.33:0.1]
        grid32 = mgrid[np.float32(0.1):np.float32(0.33):np.float32(0.1)]
        assert_(grid32.dtype == np.float64)
        assert_array_almost_equal(grid64, grid32)

    def test_accepts_longdouble(self):
        grid64 = mgrid[0.1:0.33:0.1,]
        grid128 = mgrid[np.longdouble(0.1):np.longdouble(0.33):np.longdouble(0.1),]
        assert_(grid128.dtype == np.longdouble)
        assert_array_almost_equal(grid64, grid128)
        grid128c_a = mgrid[0:np.longdouble(1):3.4j]
        grid128c_b = mgrid[0:np.longdouble(1):3.4j,]
        assert_(grid128c_a.dtype == grid128c_b.dtype == np.longdouble)
        assert_array_equal(grid128c_a, grid128c_b[0])
        grid64 = mgrid[0.1:0.33:0.1]
        grid128 = mgrid[np.longdouble(0.1):np.longdouble(0.33):np.longdouble(0.1)]
        assert_(grid128.dtype == np.longdouble)
        assert_array_almost_equal(grid64, grid128)

    def test_accepts_npcomplexfloating(self):
        assert_array_almost_equal(mgrid[0.1:0.3:3j,], mgrid[0.1:0.3:np.complex64(3j),])
        assert_array_almost_equal(mgrid[0.1:0.3:3j], mgrid[0.1:0.3:np.complex64(3j)])
        grid64_a = mgrid[0.1:0.3:3.3j]
        grid64_b = mgrid[0.1:0.3:3.3j,][0]
        assert_(grid64_a.dtype == grid64_b.dtype == np.float64)
        assert_array_equal(grid64_a, grid64_b)
        grid128_a = mgrid[0.1:0.3:np.clongdouble(3.3j)]
        grid128_b = mgrid[0.1:0.3:np.clongdouble(3.3j),][0]
        assert_(grid128_a.dtype == grid128_b.dtype == np.longdouble)
        assert_array_equal(grid64_a, grid64_b)