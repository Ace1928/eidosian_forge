import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
class TestExpm1:

    def test_expm1(self):
        assert_almost_equal(ncu.expm1(0.2), ncu.exp(0.2) - 1)
        assert_almost_equal(ncu.expm1(1e-06), ncu.exp(1e-06) - 1)

    def test_special(self):
        assert_equal(ncu.expm1(np.inf), np.inf)
        assert_equal(ncu.expm1(0.0), 0.0)
        assert_equal(ncu.expm1(-0.0), -0.0)
        assert_equal(ncu.expm1(np.inf), np.inf)
        assert_equal(ncu.expm1(-np.inf), -1.0)

    def test_complex(self):
        x = np.asarray(1e-12)
        assert_allclose(x, ncu.expm1(x))
        x = x.astype(np.complex128)
        assert_allclose(x, ncu.expm1(x))