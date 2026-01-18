from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
class TestConstants:

    def test_legdomain(self):
        assert_equal(leg.legdomain, [-1, 1])

    def test_legzero(self):
        assert_equal(leg.legzero, [0])

    def test_legone(self):
        assert_equal(leg.legone, [1])

    def test_legx(self):
        assert_equal(leg.legx, [0, 1])