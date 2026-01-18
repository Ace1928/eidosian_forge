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
class TestTrimZeros:
    a = np.array([0, 0, 1, 0, 2, 3, 4, 0])
    b = a.astype(float)
    c = a.astype(complex)
    d = a.astype(object)

    def values(self):
        attr_names = ('a', 'b', 'c', 'd')
        return (getattr(self, name) for name in attr_names)

    def test_basic(self):
        slc = np.s_[2:-1]
        for arr in self.values():
            res = trim_zeros(arr)
            assert_array_equal(res, arr[slc])

    def test_leading_skip(self):
        slc = np.s_[:-1]
        for arr in self.values():
            res = trim_zeros(arr, trim='b')
            assert_array_equal(res, arr[slc])

    def test_trailing_skip(self):
        slc = np.s_[2:]
        for arr in self.values():
            res = trim_zeros(arr, trim='F')
            assert_array_equal(res, arr[slc])

    def test_all_zero(self):
        for _arr in self.values():
            arr = np.zeros_like(_arr, dtype=_arr.dtype)
            res1 = trim_zeros(arr, trim='B')
            assert len(res1) == 0
            res2 = trim_zeros(arr, trim='f')
            assert len(res2) == 0

    def test_size_zero(self):
        arr = np.zeros(0)
        res = trim_zeros(arr)
        assert_array_equal(arr, res)

    @pytest.mark.parametrize('arr', [np.array([0, 2 ** 62, 0]), np.array([0, 2 ** 63, 0]), np.array([0, 2 ** 64, 0])])
    def test_overflow(self, arr):
        slc = np.s_[1:2]
        res = trim_zeros(arr)
        assert_array_equal(res, arr[slc])

    def test_no_trim(self):
        arr = np.array([None, 1, None])
        res = trim_zeros(arr)
        assert_array_equal(arr, res)

    def test_list_to_list(self):
        res = trim_zeros(self.a.tolist())
        assert isinstance(res, list)