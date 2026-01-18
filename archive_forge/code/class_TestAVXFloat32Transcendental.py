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
class TestAVXFloat32Transcendental:

    def test_exp_float32(self):
        np.random.seed(42)
        x_f32 = np.float32(np.random.uniform(low=0.0, high=88.1, size=1000000))
        x_f64 = np.float64(x_f32)
        assert_array_max_ulp(np.exp(x_f32), np.float32(np.exp(x_f64)), maxulp=3)

    def test_log_float32(self):
        np.random.seed(42)
        x_f32 = np.float32(np.random.uniform(low=0.0, high=1000, size=1000000))
        x_f64 = np.float64(x_f32)
        assert_array_max_ulp(np.log(x_f32), np.float32(np.log(x_f64)), maxulp=4)

    def test_sincos_float32(self):
        np.random.seed(42)
        N = 1000000
        M = np.int_(N / 20)
        index = np.random.randint(low=0, high=N, size=M)
        x_f32 = np.float32(np.random.uniform(low=-100.0, high=100.0, size=N))
        if not _glibc_older_than('2.17'):
            x_f32[index] = np.float32(100000000000.0 * np.random.rand(M))
        x_f64 = np.float64(x_f32)
        assert_array_max_ulp(np.sin(x_f32), np.float32(np.sin(x_f64)), maxulp=2)
        assert_array_max_ulp(np.cos(x_f32), np.float32(np.cos(x_f64)), maxulp=2)
        tx_f32 = x_f32.copy()
        assert_array_max_ulp(np.sin(x_f32, out=x_f32), np.float32(np.sin(x_f64)), maxulp=2)
        assert_array_max_ulp(np.cos(tx_f32, out=tx_f32), np.float32(np.cos(x_f64)), maxulp=2)

    def test_strided_float32(self):
        np.random.seed(42)
        strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
        sizes = np.arange(2, 100)
        for ii in sizes:
            x_f32 = np.float32(np.random.uniform(low=0.01, high=88.1, size=ii))
            x_f32_large = x_f32.copy()
            x_f32_large[3:-1:4] = 120000.0
            exp_true = np.exp(x_f32)
            log_true = np.log(x_f32)
            sin_true = np.sin(x_f32_large)
            cos_true = np.cos(x_f32_large)
            for jj in strides:
                assert_array_almost_equal_nulp(np.exp(x_f32[::jj]), exp_true[::jj], nulp=2)
                assert_array_almost_equal_nulp(np.log(x_f32[::jj]), log_true[::jj], nulp=2)
                assert_array_almost_equal_nulp(np.sin(x_f32_large[::jj]), sin_true[::jj], nulp=2)
                assert_array_almost_equal_nulp(np.cos(x_f32_large[::jj]), cos_true[::jj], nulp=2)