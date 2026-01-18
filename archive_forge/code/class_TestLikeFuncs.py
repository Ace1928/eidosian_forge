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
class TestLikeFuncs:
    """Test ones_like, zeros_like, empty_like and full_like"""

    def setup_method(self):
        self.data = [(np.array(3.0), None), (np.array(3), 'f8'), (np.arange(6, dtype='f4'), None), (np.arange(6), 'c16'), (np.arange(6).reshape(2, 3), None), (np.arange(6).reshape(3, 2), 'i1'), (np.arange(6).reshape((2, 3), order='F'), None), (np.arange(6).reshape((3, 2), order='F'), 'i1'), (np.arange(24).reshape(2, 3, 4), None), (np.arange(24).reshape(4, 3, 2), 'f4'), (np.arange(24).reshape((2, 3, 4), order='F'), None), (np.arange(24).reshape((4, 3, 2), order='F'), 'f4'), (np.arange(24).reshape(2, 3, 4).swapaxes(0, 1), None), (np.arange(24).reshape(4, 3, 2).swapaxes(0, 1), '?')]
        self.shapes = [(), (5,), (5, 6), (5, 6, 7)]

    def compare_array_value(self, dz, value, fill_value):
        if value is not None:
            if fill_value:
                z = np.array(value).astype(dz.dtype)
                assert_(np.all(dz == z))
            else:
                assert_(np.all(dz == value))

    def check_like_function(self, like_function, value, fill_value=False):
        if fill_value:
            fill_kwarg = {'fill_value': value}
        else:
            fill_kwarg = {}
        for d, dtype in self.data:
            dz = like_function(d, dtype=dtype, **fill_kwarg)
            assert_equal(dz.shape, d.shape)
            assert_equal(np.array(dz.strides) * d.dtype.itemsize, np.array(d.strides) * dz.dtype.itemsize)
            assert_equal(d.flags.c_contiguous, dz.flags.c_contiguous)
            assert_equal(d.flags.f_contiguous, dz.flags.f_contiguous)
            if dtype is None:
                assert_equal(dz.dtype, d.dtype)
            else:
                assert_equal(dz.dtype, np.dtype(dtype))
            self.compare_array_value(dz, value, fill_value)
            dz = like_function(d, order='C', dtype=dtype, **fill_kwarg)
            assert_equal(dz.shape, d.shape)
            assert_(dz.flags.c_contiguous)
            if dtype is None:
                assert_equal(dz.dtype, d.dtype)
            else:
                assert_equal(dz.dtype, np.dtype(dtype))
            self.compare_array_value(dz, value, fill_value)
            dz = like_function(d, order='F', dtype=dtype, **fill_kwarg)
            assert_equal(dz.shape, d.shape)
            assert_(dz.flags.f_contiguous)
            if dtype is None:
                assert_equal(dz.dtype, d.dtype)
            else:
                assert_equal(dz.dtype, np.dtype(dtype))
            self.compare_array_value(dz, value, fill_value)
            dz = like_function(d, order='A', dtype=dtype, **fill_kwarg)
            assert_equal(dz.shape, d.shape)
            if d.flags.f_contiguous:
                assert_(dz.flags.f_contiguous)
            else:
                assert_(dz.flags.c_contiguous)
            if dtype is None:
                assert_equal(dz.dtype, d.dtype)
            else:
                assert_equal(dz.dtype, np.dtype(dtype))
            self.compare_array_value(dz, value, fill_value)
            for s in self.shapes:
                for o in 'CFA':
                    sz = like_function(d, dtype=dtype, shape=s, order=o, **fill_kwarg)
                    assert_equal(sz.shape, s)
                    if dtype is None:
                        assert_equal(sz.dtype, d.dtype)
                    else:
                        assert_equal(sz.dtype, np.dtype(dtype))
                    if o == 'C' or (o == 'A' and d.flags.c_contiguous):
                        assert_(sz.flags.c_contiguous)
                    elif o == 'F' or (o == 'A' and d.flags.f_contiguous):
                        assert_(sz.flags.f_contiguous)
                    self.compare_array_value(sz, value, fill_value)
                if d.ndim != len(s):
                    assert_equal(np.argsort(like_function(d, dtype=dtype, shape=s, order='K', **fill_kwarg).strides), np.argsort(np.empty(s, dtype=dtype, order='C').strides))
                else:
                    assert_equal(np.argsort(like_function(d, dtype=dtype, shape=s, order='K', **fill_kwarg).strides), np.argsort(d.strides))

        class MyNDArray(np.ndarray):
            pass
        a = np.array([[1, 2], [3, 4]]).view(MyNDArray)
        b = like_function(a, **fill_kwarg)
        assert_(type(b) is MyNDArray)
        b = like_function(a, subok=False, **fill_kwarg)
        assert_(type(b) is not MyNDArray)

    def test_ones_like(self):
        self.check_like_function(np.ones_like, 1)

    def test_zeros_like(self):
        self.check_like_function(np.zeros_like, 0)

    def test_empty_like(self):
        self.check_like_function(np.empty_like, None)

    def test_filled_like(self):
        self.check_like_function(np.full_like, 0, True)
        self.check_like_function(np.full_like, 1, True)
        self.check_like_function(np.full_like, 1000, True)
        self.check_like_function(np.full_like, 123.456, True)
        with np.errstate(invalid='ignore'):
            self.check_like_function(np.full_like, np.inf, True)

    @pytest.mark.parametrize('likefunc', [np.empty_like, np.full_like, np.zeros_like, np.ones_like])
    @pytest.mark.parametrize('dtype', [str, bytes])
    def test_dtype_str_bytes(self, likefunc, dtype):
        a = np.arange(16).reshape(2, 8)
        b = a[:, ::2]
        kwargs = {'fill_value': ''} if likefunc == np.full_like else {}
        result = likefunc(b, dtype=dtype, **kwargs)
        if dtype == str:
            assert result.strides == (16, 4)
        else:
            assert result.strides == (4, 1)