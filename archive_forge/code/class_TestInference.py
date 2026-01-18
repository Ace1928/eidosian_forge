import collections
from collections import namedtuple
from collections.abc import Iterator
from datetime import (
from decimal import Decimal
from fractions import Fraction
from io import StringIO
import itertools
from numbers import Number
import re
import sys
from typing import (
import numpy as np
import pytest
import pytz
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes import inference
from pandas.core.dtypes.cast import find_result_type
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
class TestInference:

    @pytest.mark.parametrize('arr', [np.array(list('abc'), dtype='S1'), np.array(list('abc'), dtype='S1').astype(object), [b'a', np.nan, b'c']])
    def test_infer_dtype_bytes(self, arr):
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'bytes'

    @pytest.mark.parametrize('value, expected', [(float('inf'), True), (np.inf, True), (-np.inf, False), (1, False), ('a', False)])
    def test_isposinf_scalar(self, value, expected):
        result = libmissing.isposinf_scalar(value)
        assert result is expected

    @pytest.mark.parametrize('value, expected', [(float('-inf'), True), (-np.inf, True), (np.inf, False), (1, False), ('a', False)])
    def test_isneginf_scalar(self, value, expected):
        result = libmissing.isneginf_scalar(value)
        assert result is expected

    @pytest.mark.parametrize('convert_to_masked_nullable, exp', [(True, BooleanArray(np.array([True, False], dtype='bool'), np.array([False, True]))), (False, np.array([True, np.nan], dtype='object'))])
    def test_maybe_convert_nullable_boolean(self, convert_to_masked_nullable, exp):
        arr = np.array([True, np.nan], dtype=object)
        result = libops.maybe_convert_bool(arr, set(), convert_to_masked_nullable=convert_to_masked_nullable)
        if convert_to_masked_nullable:
            tm.assert_extension_array_equal(BooleanArray(*result), exp)
        else:
            result = result[0]
            tm.assert_numpy_array_equal(result, exp)

    @pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
    @pytest.mark.parametrize('coerce_numeric', [True, False])
    @pytest.mark.parametrize('infinity', ['inf', 'inF', 'iNf', 'Inf', 'iNF', 'InF', 'INf', 'INF'])
    @pytest.mark.parametrize('prefix', ['', '-', '+'])
    def test_maybe_convert_numeric_infinities(self, coerce_numeric, infinity, prefix, convert_to_masked_nullable):
        result, _ = lib.maybe_convert_numeric(np.array([prefix + infinity], dtype=object), na_values={'', 'NULL', 'nan'}, coerce_numeric=coerce_numeric, convert_to_masked_nullable=convert_to_masked_nullable)
        expected = np.array([np.inf if prefix in ['', '+'] else -np.inf])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
    def test_maybe_convert_numeric_infinities_raises(self, convert_to_masked_nullable):
        msg = 'Unable to parse string'
        with pytest.raises(ValueError, match=msg):
            lib.maybe_convert_numeric(np.array(['foo_inf'], dtype=object), na_values={'', 'NULL', 'nan'}, coerce_numeric=False, convert_to_masked_nullable=convert_to_masked_nullable)

    @pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
    def test_maybe_convert_numeric_post_floatify_nan(self, coerce, convert_to_masked_nullable):
        data = np.array(['1.200', '-999.000', '4.500'], dtype=object)
        expected = np.array([1.2, np.nan, 4.5], dtype=np.float64)
        nan_values = {-999, -999.0}
        out = lib.maybe_convert_numeric(data, nan_values, coerce, convert_to_masked_nullable=convert_to_masked_nullable)
        if convert_to_masked_nullable:
            expected = FloatingArray(expected, np.isnan(expected))
            tm.assert_extension_array_equal(expected, FloatingArray(*out))
        else:
            out = out[0]
            tm.assert_numpy_array_equal(out, expected)

    def test_convert_infs(self):
        arr = np.array(['inf', 'inf', 'inf'], dtype='O')
        result, _ = lib.maybe_convert_numeric(arr, set(), False)
        assert result.dtype == np.float64
        arr = np.array(['-inf', '-inf', '-inf'], dtype='O')
        result, _ = lib.maybe_convert_numeric(arr, set(), False)
        assert result.dtype == np.float64

    def test_scientific_no_exponent(self):
        arr = np.array(['42E', '2E', '99e', '6e'], dtype='O')
        result, _ = lib.maybe_convert_numeric(arr, set(), False, True)
        assert np.all(np.isnan(result))

    def test_convert_non_hashable(self):
        arr = np.array([[10.0, 2], 1.0, 'apple'], dtype=object)
        result, _ = lib.maybe_convert_numeric(arr, set(), False, True)
        tm.assert_numpy_array_equal(result, np.array([np.nan, 1.0, np.nan]))

    def test_convert_numeric_uint64(self):
        arr = np.array([2 ** 63], dtype=object)
        exp = np.array([2 ** 63], dtype=np.uint64)
        tm.assert_numpy_array_equal(lib.maybe_convert_numeric(arr, set())[0], exp)
        arr = np.array([str(2 ** 63)], dtype=object)
        exp = np.array([2 ** 63], dtype=np.uint64)
        tm.assert_numpy_array_equal(lib.maybe_convert_numeric(arr, set())[0], exp)
        arr = np.array([np.uint64(2 ** 63)], dtype=object)
        exp = np.array([2 ** 63], dtype=np.uint64)
        tm.assert_numpy_array_equal(lib.maybe_convert_numeric(arr, set())[0], exp)

    @pytest.mark.parametrize('arr', [np.array([2 ** 63, np.nan], dtype=object), np.array([str(2 ** 63), np.nan], dtype=object), np.array([np.nan, 2 ** 63], dtype=object), np.array([np.nan, str(2 ** 63)], dtype=object)])
    def test_convert_numeric_uint64_nan(self, coerce, arr):
        expected = arr.astype(float) if coerce else arr.copy()
        result, _ = lib.maybe_convert_numeric(arr, set(), coerce_numeric=coerce)
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
    def test_convert_numeric_uint64_nan_values(self, coerce, convert_to_masked_nullable):
        arr = np.array([2 ** 63, 2 ** 63 + 1], dtype=object)
        na_values = {2 ** 63}
        expected = np.array([np.nan, 2 ** 63 + 1], dtype=float) if coerce else arr.copy()
        result = lib.maybe_convert_numeric(arr, na_values, coerce_numeric=coerce, convert_to_masked_nullable=convert_to_masked_nullable)
        if convert_to_masked_nullable and coerce:
            expected = IntegerArray(np.array([0, 2 ** 63 + 1], dtype='u8'), np.array([True, False], dtype='bool'))
            result = IntegerArray(*result)
        else:
            result = result[0]
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize('case', [np.array([2 ** 63, -1], dtype=object), np.array([str(2 ** 63), -1], dtype=object), np.array([str(2 ** 63), str(-1)], dtype=object), np.array([-1, 2 ** 63], dtype=object), np.array([-1, str(2 ** 63)], dtype=object), np.array([str(-1), str(2 ** 63)], dtype=object)])
    @pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
    def test_convert_numeric_int64_uint64(self, case, coerce, convert_to_masked_nullable):
        expected = case.astype(float) if coerce else case.copy()
        result, _ = lib.maybe_convert_numeric(case, set(), coerce_numeric=coerce, convert_to_masked_nullable=convert_to_masked_nullable)
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
    def test_convert_numeric_string_uint64(self, convert_to_masked_nullable):
        result = lib.maybe_convert_numeric(np.array(['uint64'], dtype=object), set(), coerce_numeric=True, convert_to_masked_nullable=convert_to_masked_nullable)
        if convert_to_masked_nullable:
            result = FloatingArray(*result)
        else:
            result = result[0]
        assert np.isnan(result)

    @pytest.mark.parametrize('value', [-2 ** 63 - 1, 2 ** 64])
    def test_convert_int_overflow(self, value):
        arr = np.array([value], dtype=object)
        result = lib.maybe_convert_objects(arr)
        tm.assert_numpy_array_equal(arr, result)

    @pytest.mark.parametrize('val', [None, np.nan, float('nan')])
    @pytest.mark.parametrize('dtype', ['M8[ns]', 'm8[ns]'])
    def test_maybe_convert_objects_nat_inference(self, val, dtype):
        dtype = np.dtype(dtype)
        vals = np.array([pd.NaT, val], dtype=object)
        result = lib.maybe_convert_objects(vals, convert_non_numeric=True, dtype_if_all_nat=dtype)
        assert result.dtype == dtype
        assert np.isnat(result).all()
        result = lib.maybe_convert_objects(vals[::-1], convert_non_numeric=True, dtype_if_all_nat=dtype)
        assert result.dtype == dtype
        assert np.isnat(result).all()

    @pytest.mark.parametrize('value, expected_dtype', [([2 ** 63], np.uint64), ([np.uint64(2 ** 63)], np.uint64), ([2, -1], np.int64), ([2 ** 63, -1], object), ([np.uint8(1)], np.uint8), ([np.uint16(1)], np.uint16), ([np.uint32(1)], np.uint32), ([np.uint64(1)], np.uint64), ([np.uint8(2), np.uint16(1)], np.uint16), ([np.uint32(2), np.uint16(1)], np.uint32), ([np.uint32(2), -1], object), ([np.uint32(2), 1], np.uint64), ([np.uint32(2), np.int32(1)], object)])
    def test_maybe_convert_objects_uint(self, value, expected_dtype):
        arr = np.array(value, dtype=object)
        exp = np.array(value, dtype=expected_dtype)
        tm.assert_numpy_array_equal(lib.maybe_convert_objects(arr), exp)

    def test_maybe_convert_objects_datetime(self):
        arr = np.array([np.datetime64('2000-01-01'), np.timedelta64(1, 's')], dtype=object)
        exp = arr.copy()
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        tm.assert_numpy_array_equal(out, exp)
        arr = np.array([pd.NaT, np.timedelta64(1, 's')], dtype=object)
        exp = np.array([np.timedelta64('NaT'), np.timedelta64(1, 's')], dtype='m8[ns]')
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        tm.assert_numpy_array_equal(out, exp)
        arr = np.array([np.timedelta64(1, 's'), np.nan], dtype=object)
        exp = exp[::-1]
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        tm.assert_numpy_array_equal(out, exp)

    def test_maybe_convert_objects_dtype_if_all_nat(self):
        arr = np.array([pd.NaT, pd.NaT], dtype=object)
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        tm.assert_numpy_array_equal(out, arr)
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True, dtype_if_all_nat=np.dtype('timedelta64[ns]'))
        exp = np.array(['NaT', 'NaT'], dtype='timedelta64[ns]')
        tm.assert_numpy_array_equal(out, exp)
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True, dtype_if_all_nat=np.dtype('datetime64[ns]'))
        exp = np.array(['NaT', 'NaT'], dtype='datetime64[ns]')
        tm.assert_numpy_array_equal(out, exp)

    def test_maybe_convert_objects_dtype_if_all_nat_invalid(self):
        arr = np.array([pd.NaT, pd.NaT], dtype=object)
        with pytest.raises(ValueError, match='int64'):
            lib.maybe_convert_objects(arr, convert_non_numeric=True, dtype_if_all_nat=np.dtype('int64'))

    @pytest.mark.parametrize('dtype', ['datetime64[ns]', 'timedelta64[ns]'])
    def test_maybe_convert_objects_datetime_overflow_safe(self, dtype):
        stamp = datetime(2363, 10, 4)
        if dtype == 'timedelta64[ns]':
            stamp = stamp - datetime(1970, 1, 1)
        arr = np.array([stamp], dtype=object)
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        tm.assert_numpy_array_equal(out, arr)

    def test_maybe_convert_objects_mixed_datetimes(self):
        ts = Timestamp('now')
        vals = [ts, ts.to_pydatetime(), ts.to_datetime64(), pd.NaT, np.nan, None]
        for data in itertools.permutations(vals):
            data = np.array(list(data), dtype=object)
            expected = DatetimeIndex(data)._data._ndarray
            result = lib.maybe_convert_objects(data, convert_non_numeric=True)
            tm.assert_numpy_array_equal(result, expected)

    def test_maybe_convert_objects_timedelta64_nat(self):
        obj = np.timedelta64('NaT', 'ns')
        arr = np.array([obj], dtype=object)
        assert arr[0] is obj
        result = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        expected = np.array([obj], dtype='m8[ns]')
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('exp', [IntegerArray(np.array([2, 0], dtype='i8'), np.array([False, True])), IntegerArray(np.array([2, 0], dtype='int64'), np.array([False, True]))])
    def test_maybe_convert_objects_nullable_integer(self, exp):
        arr = np.array([2, np.nan], dtype=object)
        result = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
        tm.assert_extension_array_equal(result, exp)

    @pytest.mark.parametrize('dtype, val', [('int64', 1), ('uint64', np.iinfo(np.int64).max + 1)])
    def test_maybe_convert_objects_nullable_none(self, dtype, val):
        arr = np.array([val, None, 3], dtype='object')
        result = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
        expected = IntegerArray(np.array([val, 0, 3], dtype=dtype), np.array([False, True, False]))
        tm.assert_extension_array_equal(result, expected)

    @pytest.mark.parametrize('convert_to_masked_nullable, exp', [(True, IntegerArray(np.array([2, 0], dtype='i8'), np.array([False, True]))), (False, np.array([2, np.nan], dtype='float64'))])
    def test_maybe_convert_numeric_nullable_integer(self, convert_to_masked_nullable, exp):
        arr = np.array([2, np.nan], dtype=object)
        result = lib.maybe_convert_numeric(arr, set(), convert_to_masked_nullable=convert_to_masked_nullable)
        if convert_to_masked_nullable:
            result = IntegerArray(*result)
            tm.assert_extension_array_equal(result, exp)
        else:
            result = result[0]
            tm.assert_numpy_array_equal(result, exp)

    @pytest.mark.parametrize('convert_to_masked_nullable, exp', [(True, FloatingArray(np.array([2.0, 0.0], dtype='float64'), np.array([False, True]))), (False, np.array([2.0, np.nan], dtype='float64'))])
    def test_maybe_convert_numeric_floating_array(self, convert_to_masked_nullable, exp):
        arr = np.array([2.0, np.nan], dtype=object)
        result = lib.maybe_convert_numeric(arr, set(), convert_to_masked_nullable=convert_to_masked_nullable)
        if convert_to_masked_nullable:
            tm.assert_extension_array_equal(FloatingArray(*result), exp)
        else:
            result = result[0]
            tm.assert_numpy_array_equal(result, exp)

    def test_maybe_convert_objects_bool_nan(self):
        ind = Index([True, False, np.nan], dtype=object)
        exp = np.array([True, False, np.nan], dtype=object)
        out = lib.maybe_convert_objects(ind.values, safe=1)
        tm.assert_numpy_array_equal(out, exp)

    def test_maybe_convert_objects_nullable_boolean(self):
        arr = np.array([True, False], dtype=object)
        exp = np.array([True, False])
        out = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
        tm.assert_numpy_array_equal(out, exp)
        arr = np.array([True, False, pd.NaT], dtype=object)
        exp = np.array([True, False, pd.NaT], dtype=object)
        out = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
        tm.assert_numpy_array_equal(out, exp)

    @pytest.mark.parametrize('val', [None, np.nan])
    def test_maybe_convert_objects_nullable_boolean_na(self, val):
        arr = np.array([True, False, val], dtype=object)
        exp = BooleanArray(np.array([True, False, False]), np.array([False, False, True]))
        out = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
        tm.assert_extension_array_equal(out, exp)

    @pytest.mark.parametrize('data0', [True, 1, 1.0, 1.0 + 1j, np.int8(1), np.int16(1), np.int32(1), np.int64(1), np.float16(1), np.float32(1), np.float64(1), np.complex64(1), np.complex128(1)])
    @pytest.mark.parametrize('data1', [True, 1, 1.0, 1.0 + 1j, np.int8(1), np.int16(1), np.int32(1), np.int64(1), np.float16(1), np.float32(1), np.float64(1), np.complex64(1), np.complex128(1)])
    def test_maybe_convert_objects_itemsize(self, data0, data1):
        data = [data0, data1]
        arr = np.array(data, dtype='object')
        common_kind = np.result_type(type(data0), type(data1)).kind
        kind0 = 'python' if not hasattr(data0, 'dtype') else data0.dtype.kind
        kind1 = 'python' if not hasattr(data1, 'dtype') else data1.dtype.kind
        if kind0 != 'python' and kind1 != 'python':
            kind = common_kind
            itemsize = max(data0.dtype.itemsize, data1.dtype.itemsize)
        elif is_bool(data0) or is_bool(data1):
            kind = 'bool' if is_bool(data0) and is_bool(data1) else 'object'
            itemsize = ''
        elif is_complex(data0) or is_complex(data1):
            kind = common_kind
            itemsize = 16
        else:
            kind = common_kind
            itemsize = 8
        expected = np.array(data, dtype=f'{kind}{itemsize}')
        result = lib.maybe_convert_objects(arr)
        tm.assert_numpy_array_equal(result, expected)

    def test_mixed_dtypes_remain_object_array(self):
        arr = np.array([datetime(2015, 1, 1, tzinfo=pytz.utc), 1], dtype=object)
        result = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        tm.assert_numpy_array_equal(result, arr)

    @pytest.mark.parametrize('idx', [pd.IntervalIndex.from_breaks(range(5), closed='both'), pd.period_range('2016-01-01', periods=3, freq='D')])
    def test_maybe_convert_objects_ea(self, idx):
        result = lib.maybe_convert_objects(np.array(idx, dtype=object), convert_non_numeric=True)
        tm.assert_extension_array_equal(result, idx._data)