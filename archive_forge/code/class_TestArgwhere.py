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
class TestArgwhere:

    @pytest.mark.parametrize('nd', [0, 1, 2])
    def test_nd(self, nd):
        x = np.empty((2,) * nd, bool)
        x[...] = False
        assert_equal(np.argwhere(x).shape, (0, nd))
        x[...] = False
        x.flat[0] = True
        assert_equal(np.argwhere(x).shape, (1, nd))
        x[...] = True
        x.flat[0] = False
        assert_equal(np.argwhere(x).shape, (x.size - 1, nd))
        x[...] = True
        assert_equal(np.argwhere(x).shape, (x.size, nd))

    def test_2D(self):
        x = np.arange(6).reshape((2, 3))
        assert_array_equal(np.argwhere(x > 1), [[0, 2], [1, 0], [1, 1], [1, 2]])

    def test_list(self):
        assert_equal(np.argwhere([4, 0, 2, 1, 3]), [[0], [2], [3], [4]])