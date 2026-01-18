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
class _TestConvolve2d:

    def test_2d_arrays(self):
        a = [[1, 2, 3], [3, 4, 5]]
        b = [[2, 3, 4], [4, 5, 6]]
        d = array([[2, 7, 16, 17, 12], [10, 30, 62, 58, 38], [12, 31, 58, 49, 30]])
        e = convolve2d(a, b)
        assert_array_equal(e, d)

    def test_valid_mode(self):
        e = [[2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10]]
        f = [[1, 2, 3], [3, 4, 5]]
        h = array([[62, 80, 98, 116, 134]])
        g = convolve2d(e, f, 'valid')
        assert_array_equal(g, h)
        g = convolve2d(f, e, 'valid')
        assert_array_equal(g, h)

    def test_valid_mode_complx(self):
        e = [[2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10]]
        f = np.array([[1, 2, 3], [3, 4, 5]], dtype=complex) + 1j
        h = array([[62.0 + 24j, 80.0 + 30j, 98.0 + 36j, 116.0 + 42j, 134.0 + 48j]])
        g = convolve2d(e, f, 'valid')
        assert_array_almost_equal(g, h)
        g = convolve2d(f, e, 'valid')
        assert_array_equal(g, h)

    def test_fillvalue(self):
        a = [[1, 2, 3], [3, 4, 5]]
        b = [[2, 3, 4], [4, 5, 6]]
        fillval = 1
        c = convolve2d(a, b, 'full', 'fill', fillval)
        d = array([[24, 26, 31, 34, 32], [28, 40, 62, 64, 52], [32, 46, 67, 62, 48]])
        assert_array_equal(c, d)

    def test_fillvalue_errors(self):
        msg = 'could not cast `fillvalue` directly to the output '
        with np.testing.suppress_warnings() as sup:
            sup.filter(ComplexWarning, 'Casting complex values')
            with assert_raises(ValueError, match=msg):
                convolve2d([[1]], [[1, 2]], fillvalue=1j)
        msg = '`fillvalue` must be scalar or an array with '
        with assert_raises(ValueError, match=msg):
            convolve2d([[1]], [[1, 2]], fillvalue=[1, 2])

    def test_fillvalue_empty(self):
        assert_raises(ValueError, convolve2d, [[1]], [[1, 2]], fillvalue=[])

    def test_wrap_boundary(self):
        a = [[1, 2, 3], [3, 4, 5]]
        b = [[2, 3, 4], [4, 5, 6]]
        c = convolve2d(a, b, 'full', 'wrap')
        d = array([[80, 80, 74, 80, 80], [68, 68, 62, 68, 68], [80, 80, 74, 80, 80]])
        assert_array_equal(c, d)

    def test_sym_boundary(self):
        a = [[1, 2, 3], [3, 4, 5]]
        b = [[2, 3, 4], [4, 5, 6]]
        c = convolve2d(a, b, 'full', 'symm')
        d = array([[34, 30, 44, 62, 66], [52, 48, 62, 80, 84], [82, 78, 92, 110, 114]])
        assert_array_equal(c, d)

    @pytest.mark.parametrize('func', [convolve2d, correlate2d])
    @pytest.mark.parametrize('boundary, expected', [('symm', [[37.0, 42.0, 44.0, 45.0]]), ('wrap', [[43.0, 44.0, 42.0, 39.0]])])
    def test_same_with_boundary(self, func, boundary, expected):
        image = np.array([[2.0, -1.0, 3.0, 4.0]])
        kernel = np.ones((1, 21))
        result = func(image, kernel, mode='same', boundary=boundary)
        assert_array_equal(result, expected)

    def test_boundary_extension_same(self):
        import scipy.ndimage as ndi
        a = np.arange(1, 10 * 3 + 1, dtype=float).reshape(10, 3)
        b = np.arange(1, 10 * 10 + 1, dtype=float).reshape(10, 10)
        c = convolve2d(a, b, mode='same', boundary='wrap')
        assert_array_equal(c, ndi.convolve(a, b, mode='wrap', origin=(-1, -1)))

    def test_boundary_extension_full(self):
        import scipy.ndimage as ndi
        a = np.arange(1, 3 * 3 + 1, dtype=float).reshape(3, 3)
        b = np.arange(1, 6 * 6 + 1, dtype=float).reshape(6, 6)
        c = convolve2d(a, b, mode='full', boundary='wrap')
        apad = np.pad(a, ((3, 3), (3, 3)), 'wrap')
        assert_array_equal(c, ndi.convolve(apad, b, mode='wrap')[:-1, :-1])

    def test_invalid_shapes(self):
        a = np.arange(1, 7).reshape((2, 3))
        b = np.arange(-6, 0).reshape((3, 2))
        assert_raises(ValueError, convolve2d, *(a, b), **{'mode': 'valid'})
        assert_raises(ValueError, convolve2d, *(b, a), **{'mode': 'valid'})