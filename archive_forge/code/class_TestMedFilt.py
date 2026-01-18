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
class TestMedFilt:
    IN = [[50, 50, 50, 50, 50, 92, 18, 27, 65, 46], [50, 50, 50, 50, 50, 0, 72, 77, 68, 66], [50, 50, 50, 50, 50, 46, 47, 19, 64, 77], [50, 50, 50, 50, 50, 42, 15, 29, 95, 35], [50, 50, 50, 50, 50, 46, 34, 9, 21, 66], [70, 97, 28, 68, 78, 77, 61, 58, 71, 42], [64, 53, 44, 29, 68, 32, 19, 68, 24, 84], [3, 33, 53, 67, 1, 78, 74, 55, 12, 83], [7, 11, 46, 70, 60, 47, 24, 43, 61, 26], [32, 61, 88, 7, 39, 4, 92, 64, 45, 61]]
    OUT = [[0, 50, 50, 50, 42, 15, 15, 18, 27, 0], [0, 50, 50, 50, 50, 42, 19, 21, 29, 0], [50, 50, 50, 50, 50, 47, 34, 34, 46, 35], [50, 50, 50, 50, 50, 50, 42, 47, 64, 42], [50, 50, 50, 50, 50, 50, 46, 55, 64, 35], [33, 50, 50, 50, 50, 47, 46, 43, 55, 26], [32, 50, 50, 50, 50, 47, 46, 45, 55, 26], [7, 46, 50, 50, 47, 46, 46, 43, 45, 21], [0, 32, 33, 39, 32, 32, 43, 43, 43, 0], [0, 7, 11, 7, 4, 4, 19, 19, 24, 0]]
    KERNEL_SIZE = [7, 3]

    def test_basic(self):
        d = signal.medfilt(self.IN, self.KERNEL_SIZE)
        e = signal.medfilt2d(np.array(self.IN, float), self.KERNEL_SIZE)
        assert_array_equal(d, self.OUT)
        assert_array_equal(d, e)

    @pytest.mark.parametrize('dtype', [np.ubyte, np.byte, np.ushort, np.short, np_ulong, np_long, np.ulonglong, np.ulonglong, np.float32, np.float64])
    def test_types(self, dtype):
        in_typed = np.array(self.IN, dtype=dtype)
        assert_equal(signal.medfilt(in_typed).dtype, dtype)
        assert_equal(signal.medfilt2d(in_typed).dtype, dtype)

    def test_types_deprecated(self):
        dtype = np.longdouble
        in_typed = np.array(self.IN, dtype=dtype)
        msg = 'Using medfilt with arrays of dtype'
        with pytest.deprecated_call(match=msg):
            assert_equal(signal.medfilt(in_typed).dtype, dtype)
        with pytest.deprecated_call(match=msg):
            assert_equal(signal.medfilt2d(in_typed).dtype, dtype)

    @pytest.mark.parametrize('dtype', [np.bool_, np.complex64, np.complex128, np.clongdouble, np.float16])
    def test_invalid_dtypes(self, dtype):
        in_typed = np.array(self.IN, dtype=dtype)
        with pytest.raises(ValueError, match='not supported'):
            signal.medfilt(in_typed)
        with pytest.raises(ValueError, match='not supported'):
            signal.medfilt2d(in_typed)

    def test_none(self):
        msg = 'kernel_size exceeds volume.*|Using medfilt with arrays of dtype.*'
        with pytest.warns((UserWarning, DeprecationWarning), match=msg):
            assert_raises(TypeError, signal.medfilt, None)
        dummy = np.arange(10, dtype=np.float64)
        a = dummy[5:6]
        a.strides = 16
        assert_(signal.medfilt(a, 1) == 5.0)

    def test_refcounting(self):
        a = Decimal(123)
        x = np.array([a, a], dtype=object)
        if hasattr(sys, 'getrefcount'):
            n = 2 * sys.getrefcount(a)
        else:
            n = 10
        msg = 'kernel_size exceeds volume.*|Using medfilt with arrays of dtype.*'
        with pytest.warns((UserWarning, DeprecationWarning), match=msg):
            for j in range(n):
                signal.medfilt(x)
        if hasattr(sys, 'getrefcount'):
            assert_(sys.getrefcount(a) < n)
        assert_equal(x, [a, a])

    def test_object(self):
        msg = 'Using medfilt with arrays of dtype'
        with pytest.deprecated_call(match=msg):
            in_object = np.array(self.IN, dtype=object)
            out_object = np.array(self.OUT, dtype=object)
            assert_array_equal(signal.medfilt(in_object, self.KERNEL_SIZE), out_object)

    @pytest.mark.parametrize('dtype', [np.ubyte, np.float32, np.float64])
    def test_medfilt2d_parallel(self, dtype):
        in_typed = np.array(self.IN, dtype=dtype)
        expected = np.array(self.OUT, dtype=dtype)
        assert in_typed.shape == expected.shape
        M1 = expected.shape[0] // 2
        N1 = expected.shape[1] // 2
        offM = self.KERNEL_SIZE[0] // 2 + 1
        offN = self.KERNEL_SIZE[1] // 2 + 1

        def apply(chunk):
            M, N = chunk
            if M == 0:
                Min = slice(0, M1 + offM)
                Msel = slice(0, -offM)
                Mout = slice(0, M1)
            else:
                Min = slice(M1 - offM, None)
                Msel = slice(offM, None)
                Mout = slice(M1, None)
            if N == 0:
                Nin = slice(0, N1 + offN)
                Nsel = slice(0, -offN)
                Nout = slice(0, N1)
            else:
                Nin = slice(N1 - offN, None)
                Nsel = slice(offN, None)
                Nout = slice(N1, None)
            chunk_data = in_typed[Min, Nin]
            med = signal.medfilt2d(chunk_data, self.KERNEL_SIZE)
            return (med[Msel, Nsel], Mout, Nout)
        output = np.zeros_like(expected)
        with ThreadPoolExecutor(max_workers=4) as pool:
            chunks = {(0, 0), (0, 1), (1, 0), (1, 1)}
            futures = {pool.submit(apply, chunk) for chunk in chunks}
            for future in as_completed(futures):
                data, Mslice, Nslice = future.result()
                output[Mslice, Nslice] = data
        assert_array_equal(output, expected)