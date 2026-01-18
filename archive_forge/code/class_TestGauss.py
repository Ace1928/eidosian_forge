from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
class TestGauss:

    def test_100(self):
        x, w = leg.leggauss(100)
        v = leg.legvander(x, 99)
        vv = np.dot(v.T * w, v)
        vd = 1 / np.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        assert_almost_equal(vv, np.eye(100))
        tgt = 2.0
        assert_almost_equal(w.sum(), tgt)