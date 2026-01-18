import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
class TestZeroSizeArrays:

    @pytest.mark.parametrize('y', [np.zeros((10, 0, 5)), np.zeros((10, 5, 0))])
    @pytest.mark.parametrize('bc_type', ['not-a-knot', 'periodic', 'natural', 'clamped'])
    @pytest.mark.parametrize('axis', [0, 1, 2])
    @pytest.mark.parametrize('cls', [make_interp_spline, CubicSpline])
    def test_zero_size(self, cls, y, bc_type, axis):
        x = np.arange(10)
        xval = np.arange(3)
        obj = cls(x, y, bc_type=bc_type)
        assert obj(xval).size == 0
        assert obj(xval).shape == xval.shape + y.shape[1:]
        yt = np.moveaxis(y, 0, axis)
        obj = cls(x, yt, bc_type=bc_type, axis=axis)
        sh = yt.shape[:axis] + (xval.size,) + yt.shape[axis + 1:]
        assert obj(xval).size == 0
        assert obj(xval).shape == sh

    @pytest.mark.parametrize('y', [np.zeros((10, 0, 5)), np.zeros((10, 5, 0))])
    @pytest.mark.parametrize('axis', [0, 1, 2])
    @pytest.mark.parametrize('cls', [PchipInterpolator, Akima1DInterpolator])
    def test_zero_size_2(self, cls, y, axis):
        x = np.arange(10)
        xval = np.arange(3)
        obj = cls(x, y)
        assert obj(xval).size == 0
        assert obj(xval).shape == xval.shape + y.shape[1:]
        yt = np.moveaxis(y, 0, axis)
        obj = cls(x, yt, axis=axis)
        sh = yt.shape[:axis] + (xval.size,) + yt.shape[axis + 1:]
        assert obj(xval).size == 0
        assert obj(xval).shape == sh