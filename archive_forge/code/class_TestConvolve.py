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
class TestConvolve(_TestConvolve):

    def test_valid_mode2(self):
        a = [1, 2, 3, 6, 5, 3]
        b = [2, 3, 4, 5, 3, 4, 2, 2, 1]
        expected = [70, 78, 73, 65]
        out = convolve(a, b, 'valid')
        assert_array_equal(out, expected)
        out = convolve(b, a, 'valid')
        assert_array_equal(out, expected)
        a = [1 + 5j, 2 - 1j, 3 + 0j]
        b = [2 - 3j, 1 + 0j]
        expected = [2 - 3j, 8 - 10j]
        out = convolve(a, b, 'valid')
        assert_array_equal(out, expected)
        out = convolve(b, a, 'valid')
        assert_array_equal(out, expected)

    def test_same_mode(self):
        a = [1, 2, 3, 3, 1, 2]
        b = [1, 4, 3, 4, 5, 6, 7, 4, 3, 2, 1, 1, 3]
        c = convolve(a, b, 'same')
        d = array([57, 61, 63, 57, 45, 36])
        assert_array_equal(c, d)

    def test_invalid_shapes(self):
        a = np.arange(1, 7).reshape((2, 3))
        b = np.arange(-6, 0).reshape((3, 2))
        assert_raises(ValueError, convolve, *(a, b), **{'mode': 'valid'})
        assert_raises(ValueError, convolve, *(b, a), **{'mode': 'valid'})

    def test_convolve_method(self, n=100):
        types = {'uint16', 'uint64', 'int64', 'int32', 'complex128', 'float64', 'float16', 'complex64', 'float32', 'int16', 'uint8', 'uint32', 'int8', 'bool'}
        args = [(t1, t2, mode) for t1 in types for t2 in types for mode in ['valid', 'full', 'same']]
        np.random.seed(42)
        array_types = {'i': np.random.choice([0, 1], size=n), 'f': np.random.randn(n)}
        array_types['b'] = array_types['u'] = array_types['i']
        array_types['c'] = array_types['f'] + 0.5j * array_types['f']
        for t1, t2, mode in args:
            x1 = array_types[np.dtype(t1).kind].astype(t1)
            x2 = array_types[np.dtype(t2).kind].astype(t2)
            results = {key: convolve(x1, x2, method=key, mode=mode) for key in ['fft', 'direct']}
            assert_equal(results['fft'].dtype, results['direct'].dtype)
            if 'bool' in t1 and 'bool' in t2:
                assert_equal(choose_conv_method(x1, x2), 'direct')
                continue
            if any([t in {'complex64', 'float32'} for t in [t1, t2]]):
                kwargs = {'rtol': 0.0001, 'atol': 1e-06}
            elif 'float16' in [t1, t2]:
                kwargs = {'rtol': 0.001, 'atol': 0.001}
            else:
                kwargs = {'rtol': 1e-05, 'atol': 1e-08}
            assert_allclose(results['fft'], results['direct'], **kwargs)

    def test_convolve_method_large_input(self):
        for n in [10, 20, 50, 51, 52, 53, 54, 60, 62]:
            z = np.array([2 ** n], dtype=np.int64)
            fft = convolve(z, z, method='fft')
            direct = convolve(z, z, method='direct')
            if n < 50:
                assert_equal(fft, direct)
                assert_equal(fft, 2 ** (2 * n))
                assert_equal(direct, 2 ** (2 * n))

    def test_mismatched_dims(self):
        assert_raises(ValueError, convolve, [1], 2, method='direct')
        assert_raises(ValueError, convolve, 1, [2], method='direct')
        assert_raises(ValueError, convolve, [1], 2, method='fft')
        assert_raises(ValueError, convolve, 1, [2], method='fft')
        assert_raises(ValueError, convolve, [1], [[2]])
        assert_raises(ValueError, convolve, [3], 2)