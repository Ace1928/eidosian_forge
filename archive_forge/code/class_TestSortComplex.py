import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
class TestSortComplex:

    @pytest.mark.parametrize('type_in, type_out', [('l', 'D'), ('h', 'F'), ('H', 'F'), ('b', 'F'), ('B', 'F'), ('g', 'G')])
    def test_sort_real(self, type_in, type_out):
        a = np.array([5, 3, 6, 2, 1], dtype=type_in)
        actual = np.sort_complex(a)
        expected = np.sort(a).astype(type_out)
        assert_equal(actual, expected)
        assert_equal(actual.dtype, expected.dtype)

    def test_sort_complex(self):
        a = np.array([2 + 3j, 1 - 2j, 1 - 3j, 2 + 1j], dtype='D')
        expected = np.array([1 - 3j, 1 - 2j, 2 + 1j, 2 + 3j], dtype='D')
        actual = np.sort_complex(a)
        assert_equal(actual, expected)
        assert_equal(actual.dtype, expected.dtype)