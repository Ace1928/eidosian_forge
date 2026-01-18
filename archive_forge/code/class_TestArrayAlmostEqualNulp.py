import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
class TestArrayAlmostEqualNulp:

    def test_float64_pass(self):
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float64)
        x = 10 ** x
        x = np.r_[-x, x]
        eps = np.finfo(x.dtype).eps
        y = x + x * eps * nulp / 2.0
        assert_array_almost_equal_nulp(x, y, nulp)
        epsneg = np.finfo(x.dtype).epsneg
        y = x - x * epsneg * nulp / 2.0
        assert_array_almost_equal_nulp(x, y, nulp)

    def test_float64_fail(self):
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float64)
        x = 10 ** x
        x = np.r_[-x, x]
        eps = np.finfo(x.dtype).eps
        y = x + x * eps * nulp * 2.0
        assert_raises(AssertionError, assert_array_almost_equal_nulp, x, y, nulp)
        epsneg = np.finfo(x.dtype).epsneg
        y = x - x * epsneg * nulp * 2.0
        assert_raises(AssertionError, assert_array_almost_equal_nulp, x, y, nulp)

    def test_float64_ignore_nan(self):
        offset = np.uint64(4294967295)
        nan1_i64 = np.array(np.nan, dtype=np.float64).view(np.uint64)
        nan2_i64 = nan1_i64 ^ offset
        nan1_f64 = nan1_i64.view(np.float64)
        nan2_f64 = nan2_i64.view(np.float64)
        assert_array_max_ulp(nan1_f64, nan2_f64, 0)

    def test_float32_pass(self):
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float32)
        x = 10 ** x
        x = np.r_[-x, x]
        eps = np.finfo(x.dtype).eps
        y = x + x * eps * nulp / 2.0
        assert_array_almost_equal_nulp(x, y, nulp)
        epsneg = np.finfo(x.dtype).epsneg
        y = x - x * epsneg * nulp / 2.0
        assert_array_almost_equal_nulp(x, y, nulp)

    def test_float32_fail(self):
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float32)
        x = 10 ** x
        x = np.r_[-x, x]
        eps = np.finfo(x.dtype).eps
        y = x + x * eps * nulp * 2.0
        assert_raises(AssertionError, assert_array_almost_equal_nulp, x, y, nulp)
        epsneg = np.finfo(x.dtype).epsneg
        y = x - x * epsneg * nulp * 2.0
        assert_raises(AssertionError, assert_array_almost_equal_nulp, x, y, nulp)

    def test_float32_ignore_nan(self):
        offset = np.uint32(65535)
        nan1_i32 = np.array(np.nan, dtype=np.float32).view(np.uint32)
        nan2_i32 = nan1_i32 ^ offset
        nan1_f32 = nan1_i32.view(np.float32)
        nan2_f32 = nan2_i32.view(np.float32)
        assert_array_max_ulp(nan1_f32, nan2_f32, 0)

    def test_float16_pass(self):
        nulp = 5
        x = np.linspace(-4, 4, 10, dtype=np.float16)
        x = 10 ** x
        x = np.r_[-x, x]
        eps = np.finfo(x.dtype).eps
        y = x + x * eps * nulp / 2.0
        assert_array_almost_equal_nulp(x, y, nulp)
        epsneg = np.finfo(x.dtype).epsneg
        y = x - x * epsneg * nulp / 2.0
        assert_array_almost_equal_nulp(x, y, nulp)

    def test_float16_fail(self):
        nulp = 5
        x = np.linspace(-4, 4, 10, dtype=np.float16)
        x = 10 ** x
        x = np.r_[-x, x]
        eps = np.finfo(x.dtype).eps
        y = x + x * eps * nulp * 2.0
        assert_raises(AssertionError, assert_array_almost_equal_nulp, x, y, nulp)
        epsneg = np.finfo(x.dtype).epsneg
        y = x - x * epsneg * nulp * 2.0
        assert_raises(AssertionError, assert_array_almost_equal_nulp, x, y, nulp)

    def test_float16_ignore_nan(self):
        offset = np.uint16(255)
        nan1_i16 = np.array(np.nan, dtype=np.float16).view(np.uint16)
        nan2_i16 = nan1_i16 ^ offset
        nan1_f16 = nan1_i16.view(np.float16)
        nan2_f16 = nan2_i16.view(np.float16)
        assert_array_max_ulp(nan1_f16, nan2_f16, 0)

    def test_complex128_pass(self):
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float64)
        x = 10 ** x
        x = np.r_[-x, x]
        xi = x + x * 1j
        eps = np.finfo(x.dtype).eps
        y = x + x * eps * nulp / 2.0
        assert_array_almost_equal_nulp(xi, x + y * 1j, nulp)
        assert_array_almost_equal_nulp(xi, y + x * 1j, nulp)
        y = x + x * eps * nulp / 4.0
        assert_array_almost_equal_nulp(xi, y + y * 1j, nulp)
        epsneg = np.finfo(x.dtype).epsneg
        y = x - x * epsneg * nulp / 2.0
        assert_array_almost_equal_nulp(xi, x + y * 1j, nulp)
        assert_array_almost_equal_nulp(xi, y + x * 1j, nulp)
        y = x - x * epsneg * nulp / 4.0
        assert_array_almost_equal_nulp(xi, y + y * 1j, nulp)

    def test_complex128_fail(self):
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float64)
        x = 10 ** x
        x = np.r_[-x, x]
        xi = x + x * 1j
        eps = np.finfo(x.dtype).eps
        y = x + x * eps * nulp * 2.0
        assert_raises(AssertionError, assert_array_almost_equal_nulp, xi, x + y * 1j, nulp)
        assert_raises(AssertionError, assert_array_almost_equal_nulp, xi, y + x * 1j, nulp)
        y = x + x * eps * nulp
        assert_raises(AssertionError, assert_array_almost_equal_nulp, xi, y + y * 1j, nulp)
        epsneg = np.finfo(x.dtype).epsneg
        y = x - x * epsneg * nulp * 2.0
        assert_raises(AssertionError, assert_array_almost_equal_nulp, xi, x + y * 1j, nulp)
        assert_raises(AssertionError, assert_array_almost_equal_nulp, xi, y + x * 1j, nulp)
        y = x - x * epsneg * nulp
        assert_raises(AssertionError, assert_array_almost_equal_nulp, xi, y + y * 1j, nulp)

    def test_complex64_pass(self):
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float32)
        x = 10 ** x
        x = np.r_[-x, x]
        xi = x + x * 1j
        eps = np.finfo(x.dtype).eps
        y = x + x * eps * nulp / 2.0
        assert_array_almost_equal_nulp(xi, x + y * 1j, nulp)
        assert_array_almost_equal_nulp(xi, y + x * 1j, nulp)
        y = x + x * eps * nulp / 4.0
        assert_array_almost_equal_nulp(xi, y + y * 1j, nulp)
        epsneg = np.finfo(x.dtype).epsneg
        y = x - x * epsneg * nulp / 2.0
        assert_array_almost_equal_nulp(xi, x + y * 1j, nulp)
        assert_array_almost_equal_nulp(xi, y + x * 1j, nulp)
        y = x - x * epsneg * nulp / 4.0
        assert_array_almost_equal_nulp(xi, y + y * 1j, nulp)

    def test_complex64_fail(self):
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float32)
        x = 10 ** x
        x = np.r_[-x, x]
        xi = x + x * 1j
        eps = np.finfo(x.dtype).eps
        y = x + x * eps * nulp * 2.0
        assert_raises(AssertionError, assert_array_almost_equal_nulp, xi, x + y * 1j, nulp)
        assert_raises(AssertionError, assert_array_almost_equal_nulp, xi, y + x * 1j, nulp)
        y = x + x * eps * nulp
        assert_raises(AssertionError, assert_array_almost_equal_nulp, xi, y + y * 1j, nulp)
        epsneg = np.finfo(x.dtype).epsneg
        y = x - x * epsneg * nulp * 2.0
        assert_raises(AssertionError, assert_array_almost_equal_nulp, xi, x + y * 1j, nulp)
        assert_raises(AssertionError, assert_array_almost_equal_nulp, xi, y + x * 1j, nulp)
        y = x - x * epsneg * nulp
        assert_raises(AssertionError, assert_array_almost_equal_nulp, xi, y + y * 1j, nulp)