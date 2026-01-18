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
class TestHilbert:

    def test_bad_args(self):
        x = np.array([1.0 + 0j])
        assert_raises(ValueError, hilbert, x)
        x = np.arange(8.0)
        assert_raises(ValueError, hilbert, x, N=0)

    def test_hilbert_theoretical(self):
        decimal = 14
        pi = np.pi
        t = np.arange(0, 2 * pi, pi / 256)
        a0 = np.sin(t)
        a1 = np.cos(t)
        a2 = np.sin(2 * t)
        a3 = np.cos(2 * t)
        a = np.vstack([a0, a1, a2, a3])
        h = hilbert(a)
        h_abs = np.abs(h)
        h_angle = np.angle(h)
        h_real = np.real(h)
        assert_almost_equal(h_real, a, decimal)
        assert_almost_equal(h_abs, np.ones(a.shape), decimal)
        assert_almost_equal(h_angle[0, :256], np.arange(-pi / 2, pi / 2, pi / 256), decimal)
        assert_almost_equal(h_angle[1, :256], np.arange(0, pi, pi / 256), decimal)
        assert_almost_equal(h_angle[2, :128], np.arange(-pi / 2, pi / 2, pi / 128), decimal)
        assert_almost_equal(h_angle[3, :128], np.arange(0, pi, pi / 128), decimal)
        assert_almost_equal(h[1].imag, a0, decimal)

    def test_hilbert_axisN(self):
        a = np.arange(18).reshape(3, 6)
        aa = hilbert(a, axis=-1)
        assert_equal(hilbert(a.T, axis=0), aa.T)
        assert_almost_equal(hilbert(a[0]), aa[0], 14)
        aan = hilbert(a, N=20, axis=-1)
        assert_equal(aan.shape, [3, 20])
        assert_equal(hilbert(a.T, N=20, axis=0).shape, [20, 3])
        a0hilb = np.array([0.0 - 1.72015830311905j, 1.0 - 2.047794505137069j, 1.999999999999999 - 2.244055555687583j, 3.0 - 1.262750302935009j, 4.0 - 1.066489252384493j, 5.0 + 2.918022706971047j, 8.881784197001253e-17 + 3.845658908989067j, -9.444121133484362e-17 + 0.985044202202061j, -1.776356839400251e-16 + 1.332257797702019j, -3.996802888650564e-16 + 0.501905089898885j, 1.332267629550188e-16 + 0.668696078880782j, -1.192678053963799e-16 + 0.235487067862679j, -1.776356839400251e-16 + 0.286439612812121j, 3.108624468950438e-16 + 0.031676888064907j, 1.332267629550188e-16 - 0.019275656884536j, -2.360035624836702e-16 - 0.1652588660287j, 0.0 - 0.332049855010597j, 3.552713678800501e-16 - 0.403810179797771j, 8.881784197001253e-17 - 0.751023775297729j, 9.444121133484362e-17 - 0.79252210110103j])
        assert_almost_equal(aan[0], a0hilb, 14, 'N regression')

    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_hilbert_types(self, dtype):
        in_typed = np.zeros(8, dtype=dtype)
        assert_equal(np.real(signal.hilbert(in_typed)).dtype, dtype)