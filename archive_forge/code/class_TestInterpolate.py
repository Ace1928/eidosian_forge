from functools import reduce
import numpy as np
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
class TestInterpolate:

    def f(self, x):
        return x * (x - 1) * (x - 2)

    def test_raises(self):
        assert_raises(ValueError, cheb.chebinterpolate, self.f, -1)
        assert_raises(TypeError, cheb.chebinterpolate, self.f, 10.0)

    def test_dimensions(self):
        for deg in range(1, 5):
            assert_(cheb.chebinterpolate(self.f, deg).shape == (deg + 1,))

    def test_approximation(self):

        def powx(x, p):
            return x ** p
        x = np.linspace(-1, 1, 10)
        for deg in range(0, 10):
            for p in range(0, deg + 1):
                c = cheb.chebinterpolate(powx, deg, (p,))
                assert_almost_equal(cheb.chebval(x, c), powx(x, p), decimal=12)