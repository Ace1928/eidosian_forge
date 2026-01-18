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
class TestTypeInference:

    class Dummy:
        pass

    def test_inferred_dtype_fixture(self, any_skipna_inferred_dtype):
        inferred_dtype, values = any_skipna_inferred_dtype
        assert inferred_dtype == lib.infer_dtype(values, skipna=True)

    @pytest.mark.parametrize('skipna', [True, False])
    def test_length_zero(self, skipna):
        result = lib.infer_dtype(np.array([], dtype='i4'), skipna=skipna)
        assert result == 'integer'
        result = lib.infer_dtype([], skipna=skipna)
        assert result == 'empty'
        arr = np.array([np.array([], dtype=object), np.array([], dtype=object)])
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == 'empty'

    def test_integers(self):
        arr = np.array([1, 2, 3, np.int64(4), np.int32(5)], dtype='O')
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'integer'
        arr = np.array([1, 2, 3, np.int64(4), np.int32(5), 'foo'], dtype='O')
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'mixed-integer'
        arr = np.array([1, 2, 3, 4, 5], dtype='i4')
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'integer'

    @pytest.mark.parametrize('arr, skipna', [(np.array([1, 2, np.nan, np.nan, 3], dtype='O'), False), (np.array([1, 2, np.nan, np.nan, 3], dtype='O'), True), (np.array([1, 2, 3, np.int64(4), np.int32(5), np.nan], dtype='O'), False), (np.array([1, 2, 3, np.int64(4), np.int32(5), np.nan], dtype='O'), True)])
    def test_integer_na(self, arr, skipna):
        result = lib.infer_dtype(arr, skipna=skipna)
        expected = 'integer' if skipna else 'integer-na'
        assert result == expected

    def test_infer_dtype_skipna_default(self):
        arr = np.array([1, 2, 3, np.nan], dtype=object)
        result = lib.infer_dtype(arr)
        assert result == 'integer'

    def test_bools(self):
        arr = np.array([True, False, True, True, True], dtype='O')
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'boolean'
        arr = np.array([np.bool_(True), np.bool_(False)], dtype='O')
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'boolean'
        arr = np.array([True, False, True, 'foo'], dtype='O')
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'mixed'
        arr = np.array([True, False, True], dtype=bool)
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'boolean'
        arr = np.array([True, np.nan, False], dtype='O')
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'boolean'
        result = lib.infer_dtype(arr, skipna=False)
        assert result == 'mixed'

    def test_floats(self):
        arr = np.array([1.0, 2.0, 3.0, np.float64(4), np.float32(5)], dtype='O')
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'floating'
        arr = np.array([1, 2, 3, np.float64(4), np.float32(5), 'foo'], dtype='O')
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'mixed-integer'
        arr = np.array([1, 2, 3, 4, 5], dtype='f4')
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'floating'
        arr = np.array([1, 2, 3, 4, 5], dtype='f8')
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'floating'

    def test_decimals(self):
        arr = np.array([Decimal(1), Decimal(2), Decimal(3)])
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'decimal'
        arr = np.array([1.0, 2.0, Decimal(3)])
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'mixed'
        result = lib.infer_dtype(arr[::-1], skipna=True)
        assert result == 'mixed'
        arr = np.array([Decimal(1), Decimal('NaN'), Decimal(3)])
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'decimal'
        arr = np.array([Decimal(1), np.nan, Decimal(3)], dtype='O')
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'decimal'

    @pytest.mark.parametrize('skipna', [True, False])
    def test_complex(self, skipna):
        arr = np.array([1.0, 2.0, 1 + 1j])
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == 'complex'
        arr = np.array([1.0, 2.0, 1 + 1j], dtype='O')
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == 'mixed'
        result = lib.infer_dtype(arr[::-1], skipna=skipna)
        assert result == 'mixed'
        arr = np.array([1, np.nan, 1 + 1j])
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == 'complex'
        arr = np.array([1.0, np.nan, 1 + 1j], dtype='O')
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == 'mixed'
        arr = np.array([1 + 1j, np.nan, 3 + 3j], dtype='O')
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == 'complex'
        arr = np.array([1 + 1j, np.nan, 3 + 3j], dtype=np.complex64)
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == 'complex'

    def test_string(self):
        pass

    def test_unicode(self):
        arr = ['a', np.nan, 'c']
        result = lib.infer_dtype(arr, skipna=False)
        assert result == 'mixed'
        arr = ['a', np.nan, 'c']
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'string'
        arr = ['a', pd.NA, 'c']
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'string'
        arr = ['a', pd.NaT, 'c']
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'mixed'
        arr = ['a', 'c']
        result = lib.infer_dtype(arr, skipna=False)
        assert result == 'string'

    @pytest.mark.parametrize('dtype, missing, skipna, expected', [(float, np.nan, False, 'floating'), (float, np.nan, True, 'floating'), (object, np.nan, False, 'floating'), (object, np.nan, True, 'empty'), (object, None, False, 'mixed'), (object, None, True, 'empty')])
    @pytest.mark.parametrize('box', [Series, np.array])
    def test_object_empty(self, box, missing, dtype, skipna, expected):
        arr = box([missing, missing], dtype=dtype)
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == expected

    def test_datetime(self):
        dates = [datetime(2012, 1, x) for x in range(1, 20)]
        index = Index(dates)
        assert index.inferred_type == 'datetime64'

    def test_infer_dtype_datetime64(self):
        arr = np.array([np.datetime64('2011-01-01'), np.datetime64('2011-01-01')], dtype=object)
        assert lib.infer_dtype(arr, skipna=True) == 'datetime64'

    @pytest.mark.parametrize('na_value', [pd.NaT, np.nan])
    def test_infer_dtype_datetime64_with_na(self, na_value):
        arr = np.array([na_value, np.datetime64('2011-01-02')])
        assert lib.infer_dtype(arr, skipna=True) == 'datetime64'
        arr = np.array([na_value, np.datetime64('2011-01-02'), na_value])
        assert lib.infer_dtype(arr, skipna=True) == 'datetime64'

    @pytest.mark.parametrize('arr', [np.array([np.timedelta64('nat'), np.datetime64('2011-01-02')], dtype=object), np.array([np.datetime64('2011-01-02'), np.timedelta64('nat')], dtype=object), np.array([np.datetime64('2011-01-01'), Timestamp('2011-01-02')]), np.array([Timestamp('2011-01-02'), np.datetime64('2011-01-01')]), np.array([np.nan, Timestamp('2011-01-02'), 1.1]), np.array([np.nan, '2011-01-01', Timestamp('2011-01-02')], dtype=object), np.array([np.datetime64('nat'), np.timedelta64(1, 'D')], dtype=object), np.array([np.timedelta64(1, 'D'), np.datetime64('nat')], dtype=object)])
    def test_infer_datetimelike_dtype_mixed(self, arr):
        assert lib.infer_dtype(arr, skipna=False) == 'mixed'

    def test_infer_dtype_mixed_integer(self):
        arr = np.array([np.nan, Timestamp('2011-01-02'), 1])
        assert lib.infer_dtype(arr, skipna=True) == 'mixed-integer'

    @pytest.mark.parametrize('arr', [np.array([Timestamp('2011-01-01'), Timestamp('2011-01-02')]), np.array([datetime(2011, 1, 1), datetime(2012, 2, 1)]), np.array([datetime(2011, 1, 1), Timestamp('2011-01-02')])])
    def test_infer_dtype_datetime(self, arr):
        assert lib.infer_dtype(arr, skipna=True) == 'datetime'

    @pytest.mark.parametrize('na_value', [pd.NaT, np.nan])
    @pytest.mark.parametrize('time_stamp', [Timestamp('2011-01-01'), datetime(2011, 1, 1)])
    def test_infer_dtype_datetime_with_na(self, na_value, time_stamp):
        arr = np.array([na_value, time_stamp])
        assert lib.infer_dtype(arr, skipna=True) == 'datetime'
        arr = np.array([na_value, time_stamp, na_value])
        assert lib.infer_dtype(arr, skipna=True) == 'datetime'

    @pytest.mark.parametrize('arr', [np.array([Timedelta('1 days'), Timedelta('2 days')]), np.array([np.timedelta64(1, 'D'), np.timedelta64(2, 'D')], dtype=object), np.array([timedelta(1), timedelta(2)])])
    def test_infer_dtype_timedelta(self, arr):
        assert lib.infer_dtype(arr, skipna=True) == 'timedelta'

    @pytest.mark.parametrize('na_value', [pd.NaT, np.nan])
    @pytest.mark.parametrize('delta', [Timedelta('1 days'), np.timedelta64(1, 'D'), timedelta(1)])
    def test_infer_dtype_timedelta_with_na(self, na_value, delta):
        arr = np.array([na_value, delta])
        assert lib.infer_dtype(arr, skipna=True) == 'timedelta'
        arr = np.array([na_value, delta, na_value])
        assert lib.infer_dtype(arr, skipna=True) == 'timedelta'

    def test_infer_dtype_period(self):
        arr = np.array([Period('2011-01', freq='D'), Period('2011-02', freq='D')])
        assert lib.infer_dtype(arr, skipna=True) == 'period'
        arr = np.array([Period('2011-01', freq='D'), Period('2011-02', freq='M')])
        assert lib.infer_dtype(arr, skipna=True) == 'mixed'

    @pytest.mark.parametrize('klass', [pd.array, Series, Index])
    @pytest.mark.parametrize('skipna', [True, False])
    def test_infer_dtype_period_array(self, klass, skipna):
        values = klass([Period('2011-01-01', freq='D'), Period('2011-01-02', freq='D'), pd.NaT])
        assert lib.infer_dtype(values, skipna=skipna) == 'period'
        values = klass([Period('2011-01-01', freq='D'), Period('2011-01-02', freq='M'), pd.NaT])
        exp = 'unknown-array' if klass is pd.array else 'mixed'
        assert lib.infer_dtype(values, skipna=skipna) == exp

    def test_infer_dtype_period_mixed(self):
        arr = np.array([Period('2011-01', freq='M'), np.datetime64('nat')], dtype=object)
        assert lib.infer_dtype(arr, skipna=False) == 'mixed'
        arr = np.array([np.datetime64('nat'), Period('2011-01', freq='M')], dtype=object)
        assert lib.infer_dtype(arr, skipna=False) == 'mixed'

    @pytest.mark.parametrize('na_value', [pd.NaT, np.nan])
    def test_infer_dtype_period_with_na(self, na_value):
        arr = np.array([na_value, Period('2011-01', freq='D')])
        assert lib.infer_dtype(arr, skipna=True) == 'period'
        arr = np.array([na_value, Period('2011-01', freq='D'), na_value])
        assert lib.infer_dtype(arr, skipna=True) == 'period'

    def test_infer_dtype_all_nan_nat_like(self):
        arr = np.array([np.nan, np.nan])
        assert lib.infer_dtype(arr, skipna=True) == 'floating'
        arr = np.array([np.nan, np.nan, None])
        assert lib.infer_dtype(arr, skipna=True) == 'empty'
        assert lib.infer_dtype(arr, skipna=False) == 'mixed'
        arr = np.array([None, np.nan, np.nan])
        assert lib.infer_dtype(arr, skipna=True) == 'empty'
        assert lib.infer_dtype(arr, skipna=False) == 'mixed'
        arr = np.array([pd.NaT])
        assert lib.infer_dtype(arr, skipna=False) == 'datetime'
        arr = np.array([pd.NaT, np.nan])
        assert lib.infer_dtype(arr, skipna=False) == 'datetime'
        arr = np.array([np.nan, pd.NaT])
        assert lib.infer_dtype(arr, skipna=False) == 'datetime'
        arr = np.array([np.nan, pd.NaT, np.nan])
        assert lib.infer_dtype(arr, skipna=False) == 'datetime'
        arr = np.array([None, pd.NaT, None])
        assert lib.infer_dtype(arr, skipna=False) == 'datetime'
        arr = np.array([np.datetime64('nat')])
        assert lib.infer_dtype(arr, skipna=False) == 'datetime64'
        for n in [np.nan, pd.NaT, None]:
            arr = np.array([n, np.datetime64('nat'), n])
            assert lib.infer_dtype(arr, skipna=False) == 'datetime64'
            arr = np.array([pd.NaT, n, np.datetime64('nat'), n])
            assert lib.infer_dtype(arr, skipna=False) == 'datetime64'
        arr = np.array([np.timedelta64('nat')], dtype=object)
        assert lib.infer_dtype(arr, skipna=False) == 'timedelta'
        for n in [np.nan, pd.NaT, None]:
            arr = np.array([n, np.timedelta64('nat'), n])
            assert lib.infer_dtype(arr, skipna=False) == 'timedelta'
            arr = np.array([pd.NaT, n, np.timedelta64('nat'), n])
            assert lib.infer_dtype(arr, skipna=False) == 'timedelta'
        arr = np.array([pd.NaT, np.datetime64('nat'), np.timedelta64('nat'), np.nan])
        assert lib.infer_dtype(arr, skipna=False) == 'mixed'
        arr = np.array([np.timedelta64('nat'), np.datetime64('nat')], dtype=object)
        assert lib.infer_dtype(arr, skipna=False) == 'mixed'

    def test_is_datetimelike_array_all_nan_nat_like(self):
        arr = np.array([np.nan, pd.NaT, np.datetime64('nat')])
        assert lib.is_datetime_array(arr)
        assert lib.is_datetime64_array(arr)
        assert not lib.is_timedelta_or_timedelta64_array(arr)
        arr = np.array([np.nan, pd.NaT, np.timedelta64('nat')])
        assert not lib.is_datetime_array(arr)
        assert not lib.is_datetime64_array(arr)
        assert lib.is_timedelta_or_timedelta64_array(arr)
        arr = np.array([np.nan, pd.NaT, np.datetime64('nat'), np.timedelta64('nat')])
        assert not lib.is_datetime_array(arr)
        assert not lib.is_datetime64_array(arr)
        assert not lib.is_timedelta_or_timedelta64_array(arr)
        arr = np.array([np.nan, pd.NaT])
        assert lib.is_datetime_array(arr)
        assert lib.is_datetime64_array(arr)
        assert lib.is_timedelta_or_timedelta64_array(arr)
        arr = np.array([np.nan, np.nan], dtype=object)
        assert not lib.is_datetime_array(arr)
        assert not lib.is_datetime64_array(arr)
        assert not lib.is_timedelta_or_timedelta64_array(arr)
        assert lib.is_datetime_with_singletz_array(np.array([Timestamp('20130101', tz='US/Eastern'), Timestamp('20130102', tz='US/Eastern')], dtype=object))
        assert not lib.is_datetime_with_singletz_array(np.array([Timestamp('20130101', tz='US/Eastern'), Timestamp('20130102', tz='CET')], dtype=object))

    @pytest.mark.parametrize('func', ['is_datetime_array', 'is_datetime64_array', 'is_bool_array', 'is_timedelta_or_timedelta64_array', 'is_date_array', 'is_time_array', 'is_interval_array'])
    def test_other_dtypes_for_array(self, func):
        func = getattr(lib, func)
        arr = np.array(['foo', 'bar'])
        assert not func(arr)
        assert not func(arr.reshape(2, 1))
        arr = np.array([1, 2])
        assert not func(arr)
        assert not func(arr.reshape(2, 1))

    def test_date(self):
        dates = [date(2012, 1, day) for day in range(1, 20)]
        index = Index(dates)
        assert index.inferred_type == 'date'
        dates = [date(2012, 1, day) for day in range(1, 20)] + [np.nan]
        result = lib.infer_dtype(dates, skipna=False)
        assert result == 'mixed'
        result = lib.infer_dtype(dates, skipna=True)
        assert result == 'date'

    @pytest.mark.parametrize('values', [[date(2020, 1, 1), Timestamp('2020-01-01')], [Timestamp('2020-01-01'), date(2020, 1, 1)], [date(2020, 1, 1), pd.NaT], [pd.NaT, date(2020, 1, 1)]])
    @pytest.mark.parametrize('skipna', [True, False])
    def test_infer_dtype_date_order_invariant(self, values, skipna):
        result = lib.infer_dtype(values, skipna=skipna)
        assert result == 'date'

    def test_is_numeric_array(self):
        assert lib.is_float_array(np.array([1, 2.0]))
        assert lib.is_float_array(np.array([1, 2.0, np.nan]))
        assert not lib.is_float_array(np.array([1, 2]))
        assert lib.is_integer_array(np.array([1, 2]))
        assert not lib.is_integer_array(np.array([1, 2.0]))

    def test_is_string_array(self):
        assert lib.is_string_array(np.array(['foo', 'bar']))
        assert not lib.is_string_array(np.array(['foo', 'bar', pd.NA], dtype=object), skipna=False)
        assert lib.is_string_array(np.array(['foo', 'bar', pd.NA], dtype=object), skipna=True)
        assert lib.is_string_array(np.array(['foo', 'bar', None], dtype=object), skipna=True)
        assert lib.is_string_array(np.array(['foo', 'bar', np.nan], dtype=object), skipna=True)
        assert not lib.is_string_array(np.array(['foo', 'bar', pd.NaT], dtype=object), skipna=True)
        assert not lib.is_string_array(np.array(['foo', 'bar', np.datetime64('NaT')], dtype=object), skipna=True)
        assert not lib.is_string_array(np.array(['foo', 'bar', Decimal('NaN')], dtype=object), skipna=True)
        assert not lib.is_string_array(np.array(['foo', 'bar', None], dtype=object), skipna=False)
        assert not lib.is_string_array(np.array(['foo', 'bar', np.nan], dtype=object), skipna=False)
        assert not lib.is_string_array(np.array([1, 2]))

    def test_to_object_array_tuples(self):
        r = (5, 6)
        values = [r]
        lib.to_object_array_tuples(values)
        record = namedtuple('record', 'x y')
        r = record(5, 6)
        values = [r]
        lib.to_object_array_tuples(values)

    def test_object(self):
        arr = np.array([None], dtype='O')
        result = lib.infer_dtype(arr, skipna=False)
        assert result == 'mixed'
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'empty'

    def test_to_object_array_width(self):
        rows = [[1, 2, 3], [4, 5, 6]]
        expected = np.array(rows, dtype=object)
        out = lib.to_object_array(rows)
        tm.assert_numpy_array_equal(out, expected)
        expected = np.array(rows, dtype=object)
        out = lib.to_object_array(rows, min_width=1)
        tm.assert_numpy_array_equal(out, expected)
        expected = np.array([[1, 2, 3, None, None], [4, 5, 6, None, None]], dtype=object)
        out = lib.to_object_array(rows, min_width=5)
        tm.assert_numpy_array_equal(out, expected)

    def test_is_period(self):
        msg = 'is_period is deprecated and will be removed in a future version'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert lib.is_period(Period('2011-01', freq='M'))
            assert not lib.is_period(PeriodIndex(['2011-01'], freq='M'))
            assert not lib.is_period(Timestamp('2011-01'))
            assert not lib.is_period(1)
            assert not lib.is_period(np.nan)

    def test_is_interval(self):
        msg = 'is_interval is deprecated and will be removed in a future version'
        item = Interval(1, 2)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert lib.is_interval(item)
            assert not lib.is_interval(pd.IntervalIndex([item]))
            assert not lib.is_interval(pd.IntervalIndex([item])._engine)

    def test_categorical(self):
        arr = Categorical(list('abc'))
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'categorical'
        result = lib.infer_dtype(Series(arr), skipna=True)
        assert result == 'categorical'
        arr = Categorical(list('abc'), categories=['cegfab'], ordered=True)
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'categorical'
        result = lib.infer_dtype(Series(arr), skipna=True)
        assert result == 'categorical'

    @pytest.mark.parametrize('asobject', [True, False])
    def test_interval(self, asobject):
        idx = pd.IntervalIndex.from_breaks(range(5), closed='both')
        if asobject:
            idx = idx.astype(object)
        inferred = lib.infer_dtype(idx, skipna=False)
        assert inferred == 'interval'
        inferred = lib.infer_dtype(idx._data, skipna=False)
        assert inferred == 'interval'
        inferred = lib.infer_dtype(Series(idx, dtype=idx.dtype), skipna=False)
        assert inferred == 'interval'

    @pytest.mark.parametrize('value', [Timestamp(0), Timedelta(0), 0, 0.0])
    def test_interval_mismatched_closed(self, value):
        first = Interval(value, value, closed='left')
        second = Interval(value, value, closed='right')
        arr = np.array([first, first], dtype=object)
        assert lib.infer_dtype(arr, skipna=False) == 'interval'
        arr2 = np.array([first, second], dtype=object)
        assert lib.infer_dtype(arr2, skipna=False) == 'mixed'

    def test_interval_mismatched_subtype(self):
        first = Interval(0, 1, closed='left')
        second = Interval(Timestamp(0), Timestamp(1), closed='left')
        third = Interval(Timedelta(0), Timedelta(1), closed='left')
        arr = np.array([first, second])
        assert lib.infer_dtype(arr, skipna=False) == 'mixed'
        arr = np.array([second, third])
        assert lib.infer_dtype(arr, skipna=False) == 'mixed'
        arr = np.array([first, third])
        assert lib.infer_dtype(arr, skipna=False) == 'mixed'
        flt_interval = Interval(1.5, 2.5, closed='left')
        arr = np.array([first, flt_interval], dtype=object)
        assert lib.infer_dtype(arr, skipna=False) == 'interval'

    @pytest.mark.parametrize('klass', [pd.array, Series])
    @pytest.mark.parametrize('skipna', [True, False])
    @pytest.mark.parametrize('data', [['a', 'b', 'c'], ['a', 'b', pd.NA]])
    def test_string_dtype(self, data, skipna, klass, nullable_string_dtype):
        val = klass(data, dtype=nullable_string_dtype)
        inferred = lib.infer_dtype(val, skipna=skipna)
        assert inferred == 'string'

    @pytest.mark.parametrize('klass', [pd.array, Series])
    @pytest.mark.parametrize('skipna', [True, False])
    @pytest.mark.parametrize('data', [[True, False, True], [True, False, pd.NA]])
    def test_boolean_dtype(self, data, skipna, klass):
        val = klass(data, dtype='boolean')
        inferred = lib.infer_dtype(val, skipna=skipna)
        assert inferred == 'boolean'