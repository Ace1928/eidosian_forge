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
@pytest.mark.parametrize('dt', 'fdFD')
class TestSOSFilt:

    def test_rank1(self, dt):
        x = np.linspace(0, 5, 6).astype(dt)
        b = np.array([1, -1]).astype(dt)
        a = np.array([0.5, -0.5]).astype(dt)
        y_r = np.array([0, 2, 4, 6, 8, 10.0]).astype(dt)
        sos = tf2sos(b, a)
        assert_array_almost_equal(sosfilt(tf2sos(b, a), x), y_r)
        b = np.array([1, 1]).astype(dt)
        a = np.array([1, 0]).astype(dt)
        y_r = np.array([0, 1, 3, 5, 7, 9.0]).astype(dt)
        assert_array_almost_equal(sosfilt(tf2sos(b, a), x), y_r)
        b = [1, 1, 0]
        a = [1, 0, 0]
        x = np.ones(8)
        sos = np.concatenate((b, a))
        sos.shape = (1, 6)
        y = sosfilt(sos, x)
        assert_allclose(y, [1, 2, 2, 2, 2, 2, 2, 2])

    def test_rank2(self, dt):
        shape = (4, 3)
        x = np.linspace(0, np.prod(shape) - 1, np.prod(shape)).reshape(shape)
        x = x.astype(dt)
        b = np.array([1, -1]).astype(dt)
        a = np.array([0.5, 0.5]).astype(dt)
        y_r2_a0 = np.array([[0, 2, 4], [6, 4, 2], [0, 2, 4], [6, 4, 2]], dtype=dt)
        y_r2_a1 = np.array([[0, 2, 0], [6, -4, 6], [12, -10, 12], [18, -16, 18]], dtype=dt)
        y = sosfilt(tf2sos(b, a), x, axis=0)
        assert_array_almost_equal(y_r2_a0, y)
        y = sosfilt(tf2sos(b, a), x, axis=1)
        assert_array_almost_equal(y_r2_a1, y)

    def test_rank3(self, dt):
        shape = (4, 3, 2)
        x = np.linspace(0, np.prod(shape) - 1, np.prod(shape)).reshape(shape)
        b = np.array([1, -1]).astype(dt)
        a = np.array([0.5, 0.5]).astype(dt)
        y = sosfilt(tf2sos(b, a), x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                assert_array_almost_equal(y[i, j], lfilter(b, a, x[i, j]))

    def test_initial_conditions(self, dt):
        b1, a1 = signal.butter(2, 0.25, 'low')
        b2, a2 = signal.butter(2, 0.75, 'low')
        b3, a3 = signal.butter(2, 0.75, 'low')
        b = np.convolve(np.convolve(b1, b2), b3)
        a = np.convolve(np.convolve(a1, a2), a3)
        sos = np.array((np.r_[b1, a1], np.r_[b2, a2], np.r_[b3, a3]))
        x = np.random.rand(50).astype(dt)
        y_true, zi = lfilter(b, a, x[:20], zi=np.zeros(6))
        y_true = np.r_[y_true, lfilter(b, a, x[20:], zi=zi)[0]]
        assert_allclose_cast(y_true, lfilter(b, a, x))
        y_sos, zi = sosfilt(sos, x[:20], zi=np.zeros((3, 2)))
        y_sos = np.r_[y_sos, sosfilt(sos, x[20:], zi=zi)[0]]
        assert_allclose_cast(y_true, y_sos)
        zi = sosfilt_zi(sos)
        x = np.ones(8, dt)
        y, zf = sosfilt(sos, x, zi=zi)
        assert_allclose_cast(y, np.ones(8))
        assert_allclose_cast(zf, zi)
        x.shape = (1, 1) + x.shape
        assert_raises(ValueError, sosfilt, sos, x, zi=zi)
        zi_nd = zi.copy()
        zi_nd.shape = (zi.shape[0], 1, 1, zi.shape[-1])
        assert_raises(ValueError, sosfilt, sos, x, zi=zi_nd[:, :, :, [0, 1, 1]])
        y, zf = sosfilt(sos, x, zi=zi_nd)
        assert_allclose_cast(y[0, 0], np.ones(8))
        assert_allclose_cast(zf[:, 0, 0, :], zi)

    def test_initial_conditions_3d_axis1(self, dt):
        x = np.random.RandomState(159).randint(0, 5, size=(2, 15, 3))
        x = x.astype(dt)
        zpk = signal.butter(6, 0.35, output='zpk')
        sos = zpk2sos(*zpk)
        nsections = sos.shape[0]
        axis = 1
        shp = list(x.shape)
        shp[axis] = 2
        shp = [nsections] + shp
        z0 = np.zeros(shp)
        yf, zf = sosfilt(sos, x, axis=axis, zi=z0)
        y1, z1 = sosfilt(sos, x[:, :5, :], axis=axis, zi=z0)
        y2, z2 = sosfilt(sos, x[:, 5:, :], axis=axis, zi=z1)
        y = np.concatenate((y1, y2), axis=axis)
        assert_allclose_cast(y, yf, rtol=1e-10, atol=1e-13)
        assert_allclose_cast(z2, zf, rtol=1e-10, atol=1e-13)
        zi = sosfilt_zi(sos)
        zi.shape = [nsections, 1, 2, 1]
        zi = zi * x[:, 0:1, :]
        y = sosfilt(sos, x, axis=axis, zi=zi)[0]
        b, a = zpk2tf(*zpk)
        zi = lfilter_zi(b, a)
        zi.shape = [1, zi.size, 1]
        zi = zi * x[:, 0:1, :]
        y_tf = lfilter(b, a, x, axis=axis, zi=zi)[0]
        assert_allclose_cast(y, y_tf, rtol=1e-10, atol=1e-13)

    def test_bad_zi_shape(self, dt):
        x = np.empty((3, 15, 3), dt)
        sos = np.zeros((4, 6))
        zi = np.empty((4, 3, 3, 2))
        with pytest.raises(ValueError, match='should be all ones'):
            sosfilt(sos, x, zi=zi, axis=1)
        sos[:, 3] = 1.0
        with pytest.raises(ValueError, match='Invalid zi shape'):
            sosfilt(sos, x, zi=zi, axis=1)

    def test_sosfilt_zi(self, dt):
        sos = signal.butter(6, 0.2, output='sos')
        zi = sosfilt_zi(sos)
        y, zf = sosfilt(sos, np.ones(40, dt), zi=zi)
        assert_allclose_cast(zf, zi, rtol=1e-13)
        ss = np.prod(sos[:, :3].sum(axis=-1) / sos[:, 3:].sum(axis=-1))
        assert_allclose_cast(y, ss, rtol=1e-13)
        _, zf = sosfilt(sos, np.ones(40, dt), zi=zi.tolist())
        assert_allclose_cast(zf, zi, rtol=1e-13)