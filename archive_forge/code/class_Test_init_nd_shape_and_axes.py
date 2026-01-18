from scipy.fft._helper import next_fast_len, _init_nd_shape_and_axes
from numpy.testing import assert_equal
from pytest import raises as assert_raises
import pytest
import numpy as np
import sys
from scipy.conftest import (
from scipy._lib._array_api import xp_assert_close, SCIPY_DEVICE
from scipy import fft
class Test_init_nd_shape_and_axes:

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_py_0d_defaults(self, xp):
        x = xp.asarray(4)
        shape = None
        axes = None
        shape_expected = ()
        axes_expected = []
        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)
        assert shape_res == shape_expected
        assert axes_res == axes_expected

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_xp_0d_defaults(self, xp):
        x = xp.asarray(7.0)
        shape = None
        axes = None
        shape_expected = ()
        axes_expected = []
        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)
        assert shape_res == shape_expected
        assert axes_res == axes_expected

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_py_1d_defaults(self, xp):
        x = xp.asarray([1, 2, 3])
        shape = None
        axes = None
        shape_expected = (3,)
        axes_expected = [0]
        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)
        assert shape_res == shape_expected
        assert axes_res == axes_expected

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_xp_1d_defaults(self, xp):
        x = xp.arange(0, 1, 0.1)
        shape = None
        axes = None
        shape_expected = (10,)
        axes_expected = [0]
        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)
        assert shape_res == shape_expected
        assert axes_res == axes_expected

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_py_2d_defaults(self, xp):
        x = xp.asarray([[1, 2, 3, 4], [5, 6, 7, 8]])
        shape = None
        axes = None
        shape_expected = (2, 4)
        axes_expected = [0, 1]
        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)
        assert shape_res == shape_expected
        assert axes_res == axes_expected

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_xp_2d_defaults(self, xp):
        x = xp.arange(0, 1, 0.1)
        x = xp.reshape(x, (5, 2))
        shape = None
        axes = None
        shape_expected = (5, 2)
        axes_expected = [0, 1]
        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)
        assert shape_res == shape_expected
        assert axes_res == axes_expected

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_xp_5d_defaults(self, xp):
        x = xp.zeros([6, 2, 5, 3, 4])
        shape = None
        axes = None
        shape_expected = (6, 2, 5, 3, 4)
        axes_expected = [0, 1, 2, 3, 4]
        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)
        assert shape_res == shape_expected
        assert axes_res == axes_expected

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_xp_5d_set_shape(self, xp):
        x = xp.zeros([6, 2, 5, 3, 4])
        shape = [10, -1, -1, 1, 4]
        axes = None
        shape_expected = (10, 2, 5, 1, 4)
        axes_expected = [0, 1, 2, 3, 4]
        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)
        assert shape_res == shape_expected
        assert axes_res == axes_expected

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_xp_5d_set_axes(self, xp):
        x = xp.zeros([6, 2, 5, 3, 4])
        shape = None
        axes = [4, 1, 2]
        shape_expected = (4, 2, 5)
        axes_expected = [4, 1, 2]
        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)
        assert shape_res == shape_expected
        assert axes_res == axes_expected

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_xp_5d_set_shape_axes(self, xp):
        x = xp.zeros([6, 2, 5, 3, 4])
        shape = [10, -1, 2]
        axes = [1, 0, 3]
        shape_expected = (10, 6, 2)
        axes_expected = [1, 0, 3]
        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)
        assert shape_res == shape_expected
        assert axes_res == axes_expected

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_shape_axes_subset(self, xp):
        x = xp.zeros((2, 3, 4, 5))
        shape, axes = _init_nd_shape_and_axes(x, shape=(5, 5, 5), axes=None)
        assert shape == (5, 5, 5)
        assert axes == [1, 2, 3]

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_errors(self, xp):
        x = xp.zeros(1)
        with assert_raises(ValueError, match='axes must be a scalar or iterable of integers'):
            _init_nd_shape_and_axes(x, shape=None, axes=[[1, 2], [3, 4]])
        with assert_raises(ValueError, match='axes must be a scalar or iterable of integers'):
            _init_nd_shape_and_axes(x, shape=None, axes=[1.0, 2.0, 3.0, 4.0])
        with assert_raises(ValueError, match='axes exceeds dimensionality of input'):
            _init_nd_shape_and_axes(x, shape=None, axes=[1])
        with assert_raises(ValueError, match='axes exceeds dimensionality of input'):
            _init_nd_shape_and_axes(x, shape=None, axes=[-2])
        with assert_raises(ValueError, match='all axes must be unique'):
            _init_nd_shape_and_axes(x, shape=None, axes=[0, 0])
        with assert_raises(ValueError, match='shape must be a scalar or iterable of integers'):
            _init_nd_shape_and_axes(x, shape=[[1, 2], [3, 4]], axes=None)
        with assert_raises(ValueError, match='shape must be a scalar or iterable of integers'):
            _init_nd_shape_and_axes(x, shape=[1.0, 2.0, 3.0, 4.0], axes=None)
        with assert_raises(ValueError, match='when given, axes and shape arguments have to be of the same length'):
            _init_nd_shape_and_axes(xp.zeros([1, 1, 1, 1]), shape=[1, 2, 3], axes=[1])
        with assert_raises(ValueError, match='invalid number of data points \\(\\[0\\]\\) specified'):
            _init_nd_shape_and_axes(x, shape=[0], axes=None)
        with assert_raises(ValueError, match='invalid number of data points \\(\\[-2\\]\\) specified'):
            _init_nd_shape_and_axes(x, shape=-2, axes=None)