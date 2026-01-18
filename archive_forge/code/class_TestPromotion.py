import sys
import operator
import pytest
import ctypes
import gc
import types
from typing import Any
import numpy as np
import numpy.dtypes
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import create_custom_field_dtype
from numpy.testing import (
from numpy.compat import pickle
from itertools import permutations
import random
import hypothesis
from hypothesis.extra import numpy as hynp
class TestPromotion:
    """Test cases related to more complex DType promotions.  Further promotion
    tests are defined in `test_numeric.py`
    """

    @np._no_nep50_warning()
    @pytest.mark.parametrize(['other', 'expected', 'expected_weak'], [(2 ** 16 - 1, np.complex64, None), (2 ** 32 - 1, np.complex128, np.complex64), (np.float16(2), np.complex64, None), (np.float32(2), np.complex64, None), (np.longdouble(2), np.complex64, np.clongdouble), (np.longdouble(np.nextafter(1.7e+308, 0.0)), np.complex128, np.clongdouble), (np.longdouble(np.nextafter(1.7e+308, np.inf)), np.clongdouble, None), (np.complex64(2), np.complex64, None), (np.clongdouble(2), np.complex64, np.clongdouble), (np.clongdouble(np.nextafter(1.7e+308, 0.0) * 1j), np.complex128, np.clongdouble), (np.clongdouble(np.nextafter(1.7e+308, np.inf)), np.clongdouble, None)])
    def test_complex_other_value_based(self, weak_promotion, other, expected, expected_weak):
        if weak_promotion and expected_weak is not None:
            expected = expected_weak
        min_complex = np.dtype(np.complex64)
        res = np.result_type(other, min_complex)
        assert res == expected
        res = np.minimum(other, np.ones(3, dtype=min_complex)).dtype
        assert res == expected

    @pytest.mark.parametrize(['other', 'expected'], [(np.bool_, np.complex128), (np.int64, np.complex128), (np.float16, np.complex64), (np.float32, np.complex64), (np.float64, np.complex128), (np.longdouble, np.clongdouble), (np.complex64, np.complex64), (np.complex128, np.complex128), (np.clongdouble, np.clongdouble)])
    def test_complex_scalar_value_based(self, other, expected):
        complex_scalar = 1j
        res = np.result_type(other, complex_scalar)
        assert res == expected
        res = np.minimum(np.ones(3, dtype=other), complex_scalar).dtype
        assert res == expected

    def test_complex_pyscalar_promote_rational(self):
        with pytest.raises(TypeError, match='.* no common DType exists for the given inputs'):
            np.result_type(1j, rational)
        with pytest.raises(TypeError, match='.* no common DType exists for the given inputs'):
            np.result_type(1j, rational(1, 2))

    @pytest.mark.parametrize('val', [2, 2 ** 32, 2 ** 63, 2 ** 64, 2 * 100])
    def test_python_integer_promotion(self, val):
        expected_dtype = np.result_type(np.array(val).dtype, np.array(0).dtype)
        assert np.result_type(val, 0) == expected_dtype
        assert np.result_type(val, np.int8(0)) == expected_dtype

    @pytest.mark.parametrize(['other', 'expected'], [(1, rational), (1.0, np.float64)])
    @np._no_nep50_warning()
    def test_float_int_pyscalar_promote_rational(self, weak_promotion, other, expected):
        if not weak_promotion and type(other) == float:
            with pytest.raises(TypeError, match='.* do not have a common DType'):
                np.result_type(other, rational)
        else:
            assert np.result_type(other, rational) == expected
        assert np.result_type(other, rational(1, 2)) == expected

    @pytest.mark.parametrize(['dtypes', 'expected'], [([np.uint16, np.int16, np.float16], np.float32), ([np.uint16, np.int8, np.float16], np.float32), ([np.uint8, np.int16, np.float16], np.float32), ([1, 1, np.float64], np.float64), ([1, 1.0, np.complex128], np.complex128), ([1, 1j, np.float64], np.complex128), ([1.0, 1.0, np.int64], np.float64), ([1.0, 1j, np.float64], np.complex128), ([1j, 1j, np.float64], np.complex128), ([1, True, np.bool_], np.int_)])
    def test_permutations_do_not_influence_result(self, dtypes, expected):
        for perm in permutations(dtypes):
            assert np.result_type(*perm) == expected