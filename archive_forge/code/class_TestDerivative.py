from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
class TestDerivative:

    def test_legder(self):
        assert_raises(TypeError, leg.legder, [0], 0.5)
        assert_raises(ValueError, leg.legder, [0], -1)
        for i in range(5):
            tgt = [0] * i + [1]
            res = leg.legder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = leg.legder(leg.legint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = leg.legder(leg.legint(tgt, m=j, scl=2), m=j, scl=0.5)
                assert_almost_equal(trim(res), trim(tgt))

    def test_legder_axis(self):
        c2d = np.random.random((3, 4))
        tgt = np.vstack([leg.legder(c) for c in c2d.T]).T
        res = leg.legder(c2d, axis=0)
        assert_almost_equal(res, tgt)
        tgt = np.vstack([leg.legder(c) for c in c2d])
        res = leg.legder(c2d, axis=1)
        assert_almost_equal(res, tgt)

    def test_legder_orderhigherthancoeff(self):
        c = (1, 2, 3, 4)
        assert_equal(leg.legder(c, 4), [0])