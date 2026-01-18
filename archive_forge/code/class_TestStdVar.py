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
class TestStdVar:

    def setup_method(self):
        self.A = np.array([1, -1, 1, -1])
        self.real_var = 1

    def test_basic(self):
        assert_almost_equal(np.var(self.A), self.real_var)
        assert_almost_equal(np.std(self.A) ** 2, self.real_var)

    def test_scalars(self):
        assert_equal(np.var(1), 0)
        assert_equal(np.std(1), 0)

    def test_ddof1(self):
        assert_almost_equal(np.var(self.A, ddof=1), self.real_var * len(self.A) / (len(self.A) - 1))
        assert_almost_equal(np.std(self.A, ddof=1) ** 2, self.real_var * len(self.A) / (len(self.A) - 1))

    def test_ddof2(self):
        assert_almost_equal(np.var(self.A, ddof=2), self.real_var * len(self.A) / (len(self.A) - 2))
        assert_almost_equal(np.std(self.A, ddof=2) ** 2, self.real_var * len(self.A) / (len(self.A) - 2))

    def test_out_scalar(self):
        d = np.arange(10)
        out = np.array(0.0)
        r = np.std(d, out=out)
        assert_(r is out)
        assert_array_equal(r, out)
        r = np.var(d, out=out)
        assert_(r is out)
        assert_array_equal(r, out)
        r = np.mean(d, out=out)
        assert_(r is out)
        assert_array_equal(r, out)