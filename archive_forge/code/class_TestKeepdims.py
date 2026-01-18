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
class TestKeepdims:

    class sub_array(np.ndarray):

        def sum(self, axis=None, dtype=None, out=None):
            return np.ndarray.sum(self, axis, dtype, out, keepdims=True)

    def test_raise(self):
        sub_class = self.sub_array
        x = np.arange(30).view(sub_class)
        assert_raises(TypeError, np.sum, x, keepdims=True)