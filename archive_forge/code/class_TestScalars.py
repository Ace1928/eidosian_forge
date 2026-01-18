from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
class TestScalars:

    def test_byte(self):
        s = readsav(path.join(DATA_PATH, 'scalar_byte.sav'), verbose=False)
        assert_identical(s.i8u, np.uint8(234))

    def test_int16(self):
        s = readsav(path.join(DATA_PATH, 'scalar_int16.sav'), verbose=False)
        assert_identical(s.i16s, np.int16(-23456))

    def test_int32(self):
        s = readsav(path.join(DATA_PATH, 'scalar_int32.sav'), verbose=False)
        assert_identical(s.i32s, np.int32(-1234567890))

    def test_float32(self):
        s = readsav(path.join(DATA_PATH, 'scalar_float32.sav'), verbose=False)
        assert_identical(s.f32, np.float32(-3.1234567e+37))

    def test_float64(self):
        s = readsav(path.join(DATA_PATH, 'scalar_float64.sav'), verbose=False)
        assert_identical(s.f64, np.float64(-1.1976931348623156e+307))

    def test_complex32(self):
        s = readsav(path.join(DATA_PATH, 'scalar_complex32.sav'), verbose=False)
        assert_identical(s.c32, np.complex64(31244420000000.0 - 2.312442e+31j))

    def test_bytes(self):
        s = readsav(path.join(DATA_PATH, 'scalar_string.sav'), verbose=False)
        msg = 'The quick brown fox jumps over the lazy python'
        assert_identical(s.s, np.bytes_(msg))

    def test_structure(self):
        pass

    def test_complex64(self):
        s = readsav(path.join(DATA_PATH, 'scalar_complex64.sav'), verbose=False)
        assert_identical(s.c64, np.complex128(1.1987253647623157e+112 - 5.198725888772916e+307j))

    def test_heap_pointer(self):
        pass

    def test_object_reference(self):
        pass

    def test_uint16(self):
        s = readsav(path.join(DATA_PATH, 'scalar_uint16.sav'), verbose=False)
        assert_identical(s.i16u, np.uint16(65511))

    def test_uint32(self):
        s = readsav(path.join(DATA_PATH, 'scalar_uint32.sav'), verbose=False)
        assert_identical(s.i32u, np.uint32(4294967233))

    def test_int64(self):
        s = readsav(path.join(DATA_PATH, 'scalar_int64.sav'), verbose=False)
        assert_identical(s.i64s, np.int64(-9223372036854774567))

    def test_uint64(self):
        s = readsav(path.join(DATA_PATH, 'scalar_uint64.sav'), verbose=False)
        assert_identical(s.i64u, np.uint64(18446744073709529285))