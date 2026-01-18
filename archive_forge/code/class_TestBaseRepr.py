import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
class TestBaseRepr:

    def test_base3(self):
        assert_equal(np.base_repr(3 ** 5, 3), '100000')

    def test_positive(self):
        assert_equal(np.base_repr(12, 10), '12')
        assert_equal(np.base_repr(12, 10, 4), '000012')
        assert_equal(np.base_repr(12, 4), '30')
        assert_equal(np.base_repr(3731624803700888, 36), '10QR0ROFCEW')

    def test_negative(self):
        assert_equal(np.base_repr(-12, 10), '-12')
        assert_equal(np.base_repr(-12, 10, 4), '-000012')
        assert_equal(np.base_repr(-12, 4), '-30')

    def test_base_range(self):
        with assert_raises(ValueError):
            np.base_repr(1, 1)
        with assert_raises(ValueError):
            np.base_repr(1, 37)