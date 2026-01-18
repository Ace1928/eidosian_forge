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
class _TestConvolve:

    def test_basic(self):
        a = [3, 4, 5, 6, 5, 4]
        b = [1, 2, 3]
        c = convolve(a, b)
        assert_array_equal(c, array([3, 10, 22, 28, 32, 32, 23, 12]))

    def test_same(self):
        a = [3, 4, 5]
        b = [1, 2, 3, 4]
        c = convolve(a, b, mode='same')
        assert_array_equal(c, array([10, 22, 34]))

    def test_same_eq(self):
        a = [3, 4, 5]
        b = [1, 2, 3]
        c = convolve(a, b, mode='same')
        assert_array_equal(c, array([10, 22, 22]))

    def test_complex(self):
        x = array([1 + 1j, 2 + 1j, 3 + 1j])
        y = array([1 + 1j, 2 + 1j])
        z = convolve(x, y)
        assert_array_equal(z, array([2j, 2 + 6j, 5 + 8j, 5 + 5j]))

    def test_zero_rank(self):
        a = 1289
        b = 4567
        c = convolve(a, b)
        assert_equal(c, a * b)

    def test_broadcastable(self):
        a = np.arange(27).reshape(3, 3, 3)
        b = np.arange(3)
        for i in range(3):
            b_shape = [1] * 3
            b_shape[i] = 3
            x = convolve(a, b.reshape(b_shape), method='direct')
            y = convolve(a, b.reshape(b_shape), method='fft')
            assert_allclose(x, y)

    def test_single_element(self):
        a = array([4967])
        b = array([3920])
        c = convolve(a, b)
        assert_equal(c, a * b)

    def test_2d_arrays(self):
        a = [[1, 2, 3], [3, 4, 5]]
        b = [[2, 3, 4], [4, 5, 6]]
        c = convolve(a, b)
        d = array([[2, 7, 16, 17, 12], [10, 30, 62, 58, 38], [12, 31, 58, 49, 30]])
        assert_array_equal(c, d)

    def test_input_swapping(self):
        small = arange(8).reshape(2, 2, 2)
        big = 1j * arange(27).reshape(3, 3, 3)
        big += arange(27)[::-1].reshape(3, 3, 3)
        out_array = array([[[0 + 0j, 26 + 0j, 25 + 1j, 24 + 2j], [52 + 0j, 151 + 5j, 145 + 11j, 93 + 11j], [46 + 6j, 133 + 23j, 127 + 29j, 81 + 23j], [40 + 12j, 98 + 32j, 93 + 37j, 54 + 24j]], [[104 + 0j, 247 + 13j, 237 + 23j, 135 + 21j], [282 + 30j, 632 + 96j, 604 + 124j, 330 + 86j], [246 + 66j, 548 + 180j, 520 + 208j, 282 + 134j], [142 + 66j, 307 + 161j, 289 + 179j, 153 + 107j]], [[68 + 36j, 157 + 103j, 147 + 113j, 81 + 75j], [174 + 138j, 380 + 348j, 352 + 376j, 186 + 230j], [138 + 174j, 296 + 432j, 268 + 460j, 138 + 278j], [70 + 138j, 145 + 323j, 127 + 341j, 63 + 197j]], [[32 + 72j, 68 + 166j, 59 + 175j, 30 + 100j], [68 + 192j, 139 + 433j, 117 + 455j, 57 + 255j], [38 + 222j, 73 + 499j, 51 + 521j, 21 + 291j], [12 + 144j, 20 + 318j, 7 + 331j, 0 + 182j]]])
        assert_array_equal(convolve(small, big, 'full'), out_array)
        assert_array_equal(convolve(big, small, 'full'), out_array)
        assert_array_equal(convolve(small, big, 'same'), out_array[1:3, 1:3, 1:3])
        assert_array_equal(convolve(big, small, 'same'), out_array[0:3, 0:3, 0:3])
        assert_array_equal(convolve(small, big, 'valid'), out_array[1:3, 1:3, 1:3])
        assert_array_equal(convolve(big, small, 'valid'), out_array[1:3, 1:3, 1:3])

    def test_invalid_params(self):
        a = [3, 4, 5]
        b = [1, 2, 3]
        assert_raises(ValueError, convolve, a, b, mode='spam')
        assert_raises(ValueError, convolve, a, b, mode='eggs', method='fft')
        assert_raises(ValueError, convolve, a, b, mode='ham', method='direct')
        assert_raises(ValueError, convolve, a, b, mode='full', method='bacon')
        assert_raises(ValueError, convolve, a, b, mode='same', method='bacon')