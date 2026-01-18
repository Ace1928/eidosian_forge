from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
class TestCompanion:

    def test_raises(self):
        assert_raises(ValueError, leg.legcompanion, [])
        assert_raises(ValueError, leg.legcompanion, [1])

    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0] * i + [1]
            assert_(leg.legcompanion(coef).shape == (i, i))

    def test_linear_root(self):
        assert_(leg.legcompanion([1, 2])[0, 0] == -0.5)