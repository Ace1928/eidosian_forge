import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from itertools import product
from math import gcd
import pytest
from pytest import raises as assert_raises
from numpy.testing import (
from numpy import array, arange
import numpy as np
from scipy.fft import fft
from scipy.ndimage import correlate1d
from scipy.optimize import fmin, linear_sum_assignment
from scipy import signal
from scipy.signal import (
from scipy.signal.windows import hann
from scipy.signal._signaltools import (_filtfilt_gust, _compute_factors,
from scipy.signal._upfirdn import _upfirdn_modes
from scipy._lib import _testutils
from scipy._lib._util import ComplexWarning, np_long, np_ulong
class TestOAConvolve:

    @pytest.mark.slow()
    @pytest.mark.parametrize('shape_a_0, shape_b_0', gen_oa_shapes_eq(list(range(100)) + list(range(100, 1000, 23))))
    def test_real_manylens(self, shape_a_0, shape_b_0):
        a = np.random.rand(shape_a_0)
        b = np.random.rand(shape_b_0)
        expected = fftconvolve(a, b)
        out = oaconvolve(a, b)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('shape_a_0, shape_b_0', gen_oa_shapes([50, 47, 6, 4, 1]))
    @pytest.mark.parametrize('is_complex', [True, False])
    @pytest.mark.parametrize('mode', ['full', 'valid', 'same'])
    def test_1d_noaxes(self, shape_a_0, shape_b_0, is_complex, mode, monkeypatch):
        a = np.random.rand(shape_a_0)
        b = np.random.rand(shape_b_0)
        if is_complex:
            a = a + 1j * np.random.rand(shape_a_0)
            b = b + 1j * np.random.rand(shape_b_0)
        expected = fftconvolve(a, b, mode=mode)
        monkeypatch.setattr(signal._signaltools, 'fftconvolve', fftconvolve_err)
        out = oaconvolve(a, b, mode=mode)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [0, 1])
    @pytest.mark.parametrize('shape_a_0, shape_b_0', gen_oa_shapes([50, 47, 6, 4]))
    @pytest.mark.parametrize('shape_a_extra', [1, 3])
    @pytest.mark.parametrize('shape_b_extra', [1, 3])
    @pytest.mark.parametrize('is_complex', [True, False])
    @pytest.mark.parametrize('mode', ['full', 'valid', 'same'])
    def test_1d_axes(self, axes, shape_a_0, shape_b_0, shape_a_extra, shape_b_extra, is_complex, mode, monkeypatch):
        ax_a = [shape_a_extra] * 2
        ax_b = [shape_b_extra] * 2
        ax_a[axes] = shape_a_0
        ax_b[axes] = shape_b_0
        a = np.random.rand(*ax_a)
        b = np.random.rand(*ax_b)
        if is_complex:
            a = a + 1j * np.random.rand(*ax_a)
            b = b + 1j * np.random.rand(*ax_b)
        expected = fftconvolve(a, b, mode=mode, axes=axes)
        monkeypatch.setattr(signal._signaltools, 'fftconvolve', fftconvolve_err)
        out = oaconvolve(a, b, mode=mode, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('shape_a_0, shape_b_0, shape_a_1, shape_b_1, mode', gen_oa_shapes_2d([50, 47, 6, 4]))
    @pytest.mark.parametrize('is_complex', [True, False])
    def test_2d_noaxes(self, shape_a_0, shape_b_0, shape_a_1, shape_b_1, mode, is_complex, monkeypatch):
        a = np.random.rand(shape_a_0, shape_a_1)
        b = np.random.rand(shape_b_0, shape_b_1)
        if is_complex:
            a = a + 1j * np.random.rand(shape_a_0, shape_a_1)
            b = b + 1j * np.random.rand(shape_b_0, shape_b_1)
        expected = fftconvolve(a, b, mode=mode)
        monkeypatch.setattr(signal._signaltools, 'fftconvolve', fftconvolve_err)
        out = oaconvolve(a, b, mode=mode)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [[0, 1], [0, 2], [1, 2]])
    @pytest.mark.parametrize('shape_a_0, shape_b_0, shape_a_1, shape_b_1, mode', gen_oa_shapes_2d([50, 47, 6, 4]))
    @pytest.mark.parametrize('shape_a_extra', [1, 3])
    @pytest.mark.parametrize('shape_b_extra', [1, 3])
    @pytest.mark.parametrize('is_complex', [True, False])
    def test_2d_axes(self, axes, shape_a_0, shape_b_0, shape_a_1, shape_b_1, mode, shape_a_extra, shape_b_extra, is_complex, monkeypatch):
        ax_a = [shape_a_extra] * 3
        ax_b = [shape_b_extra] * 3
        ax_a[axes[0]] = shape_a_0
        ax_b[axes[0]] = shape_b_0
        ax_a[axes[1]] = shape_a_1
        ax_b[axes[1]] = shape_b_1
        a = np.random.rand(*ax_a)
        b = np.random.rand(*ax_b)
        if is_complex:
            a = a + 1j * np.random.rand(*ax_a)
            b = b + 1j * np.random.rand(*ax_b)
        expected = fftconvolve(a, b, mode=mode, axes=axes)
        monkeypatch.setattr(signal._signaltools, 'fftconvolve', fftconvolve_err)
        out = oaconvolve(a, b, mode=mode, axes=axes)
        assert_array_almost_equal(out, expected)

    def test_empty(self):
        assert_(oaconvolve([], []).size == 0)
        assert_(oaconvolve([5, 6], []).size == 0)
        assert_(oaconvolve([], [7]).size == 0)

    def test_zero_rank(self):
        a = array(4967)
        b = array(3920)
        out = oaconvolve(a, b)
        assert_equal(out, a * b)

    def test_single_element(self):
        a = array([4967])
        b = array([3920])
        out = oaconvolve(a, b)
        assert_equal(out, a * b)