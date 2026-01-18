from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
class TestEvaluation:
    c1d = np.array([2.0, 2.0, 2.0])
    c2d = np.einsum('i,j->ij', c1d, c1d)
    c3d = np.einsum('i,j,k->ijk', c1d, c1d, c1d)
    x = np.random.random((3, 5)) * 2 - 1
    y = polyval(x, [1.0, 2.0, 3.0])

    def test_legval(self):
        assert_equal(leg.legval([], [1]).size, 0)
        x = np.linspace(-1, 1)
        y = [polyval(x, c) for c in Llist]
        for i in range(10):
            msg = f'At i={i}'
            tgt = y[i]
            res = leg.legval(x, [0] * i + [1])
            assert_almost_equal(res, tgt, err_msg=msg)
        for i in range(3):
            dims = [2] * i
            x = np.zeros(dims)
            assert_equal(leg.legval(x, [1]).shape, dims)
            assert_equal(leg.legval(x, [1, 0]).shape, dims)
            assert_equal(leg.legval(x, [1, 0, 0]).shape, dims)

    def test_legval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y
        assert_raises(ValueError, leg.legval2d, x1, x2[:2], self.c2d)
        tgt = y1 * y2
        res = leg.legval2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)
        z = np.ones((2, 3))
        res = leg.legval2d(z, z, self.c2d)
        assert_(res.shape == (2, 3))

    def test_legval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y
        assert_raises(ValueError, leg.legval3d, x1, x2, x3[:2], self.c3d)
        tgt = y1 * y2 * y3
        res = leg.legval3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)
        z = np.ones((2, 3))
        res = leg.legval3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3))

    def test_leggrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y
        tgt = np.einsum('i,j->ij', y1, y2)
        res = leg.leggrid2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)
        z = np.ones((2, 3))
        res = leg.leggrid2d(z, z, self.c2d)
        assert_(res.shape == (2, 3) * 2)

    def test_leggrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y
        tgt = np.einsum('i,j,k->ijk', y1, y2, y3)
        res = leg.leggrid3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)
        z = np.ones((2, 3))
        res = leg.leggrid3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3) * 3)