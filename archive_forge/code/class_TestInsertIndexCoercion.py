from __future__ import annotations
from datetime import (
import itertools
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas as pd
import pandas._testing as tm
class TestInsertIndexCoercion(CoercionBase):
    klasses = ['index']
    method = 'insert'

    def _assert_insert_conversion(self, original, value, expected, expected_dtype):
        """test coercion triggered by insert"""
        target = original.copy()
        res = target.insert(1, value)
        tm.assert_index_equal(res, expected)
        assert res.dtype == expected_dtype

    @pytest.mark.parametrize('insert, coerced_val, coerced_dtype', [(1, 1, object), (1.1, 1.1, object), (False, False, object), ('x', 'x', object)])
    def test_insert_index_object(self, insert, coerced_val, coerced_dtype):
        obj = pd.Index(list('abcd'), dtype=object)
        assert obj.dtype == object
        exp = pd.Index(['a', coerced_val, 'b', 'c', 'd'], dtype=object)
        self._assert_insert_conversion(obj, insert, exp, coerced_dtype)

    @pytest.mark.parametrize('insert, coerced_val, coerced_dtype', [(1, 1, None), (1.1, 1.1, np.float64), (False, False, object), ('x', 'x', object)])
    def test_insert_int_index(self, any_int_numpy_dtype, insert, coerced_val, coerced_dtype):
        dtype = any_int_numpy_dtype
        obj = pd.Index([1, 2, 3, 4], dtype=dtype)
        coerced_dtype = coerced_dtype if coerced_dtype is not None else dtype
        exp = pd.Index([1, coerced_val, 2, 3, 4], dtype=coerced_dtype)
        self._assert_insert_conversion(obj, insert, exp, coerced_dtype)

    @pytest.mark.parametrize('insert, coerced_val, coerced_dtype', [(1, 1.0, None), (1.1, 1.1, np.float64), (False, False, object), ('x', 'x', object)])
    def test_insert_float_index(self, float_numpy_dtype, insert, coerced_val, coerced_dtype):
        dtype = float_numpy_dtype
        obj = pd.Index([1.0, 2.0, 3.0, 4.0], dtype=dtype)
        coerced_dtype = coerced_dtype if coerced_dtype is not None else dtype
        if np_version_gt2 and dtype == 'float32' and (coerced_val == 1.1):
            coerced_dtype = np.float32
        exp = pd.Index([1.0, coerced_val, 2.0, 3.0, 4.0], dtype=coerced_dtype)
        self._assert_insert_conversion(obj, insert, exp, coerced_dtype)

    @pytest.mark.parametrize('fill_val,exp_dtype', [(pd.Timestamp('2012-01-01'), 'datetime64[ns]'), (pd.Timestamp('2012-01-01', tz='US/Eastern'), 'datetime64[ns, US/Eastern]')], ids=['datetime64', 'datetime64tz'])
    @pytest.mark.parametrize('insert_value', [pd.Timestamp('2012-01-01'), pd.Timestamp('2012-01-01', tz='Asia/Tokyo'), 1])
    def test_insert_index_datetimes(self, fill_val, exp_dtype, insert_value):
        obj = pd.DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03', '2011-01-04'], tz=fill_val.tz).as_unit('ns')
        assert obj.dtype == exp_dtype
        exp = pd.DatetimeIndex(['2011-01-01', fill_val.date(), '2011-01-02', '2011-01-03', '2011-01-04'], tz=fill_val.tz).as_unit('ns')
        self._assert_insert_conversion(obj, fill_val, exp, exp_dtype)
        if fill_val.tz:
            ts = pd.Timestamp('2012-01-01')
            result = obj.insert(1, ts)
            expected = obj.astype(object).insert(1, ts)
            assert expected.dtype == object
            tm.assert_index_equal(result, expected)
            ts = pd.Timestamp('2012-01-01', tz='Asia/Tokyo')
            result = obj.insert(1, ts)
            expected = obj.insert(1, ts.tz_convert(obj.dtype.tz))
            assert expected.dtype == obj.dtype
            tm.assert_index_equal(result, expected)
        else:
            ts = pd.Timestamp('2012-01-01', tz='Asia/Tokyo')
            result = obj.insert(1, ts)
            expected = obj.astype(object).insert(1, ts)
            assert expected.dtype == object
            tm.assert_index_equal(result, expected)
        item = 1
        result = obj.insert(1, item)
        expected = obj.astype(object).insert(1, item)
        assert expected[1] == item
        assert expected.dtype == object
        tm.assert_index_equal(result, expected)

    def test_insert_index_timedelta64(self):
        obj = pd.TimedeltaIndex(['1 day', '2 day', '3 day', '4 day'])
        assert obj.dtype == 'timedelta64[ns]'
        exp = pd.TimedeltaIndex(['1 day', '10 day', '2 day', '3 day', '4 day'])
        self._assert_insert_conversion(obj, pd.Timedelta('10 day'), exp, 'timedelta64[ns]')
        for item in [pd.Timestamp('2012-01-01'), 1]:
            result = obj.insert(1, item)
            expected = obj.astype(object).insert(1, item)
            assert expected.dtype == object
            tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('insert, coerced_val, coerced_dtype', [(pd.Period('2012-01', freq='M'), '2012-01', 'period[M]'), (pd.Timestamp('2012-01-01'), pd.Timestamp('2012-01-01'), object), (1, 1, object), ('x', 'x', object)])
    def test_insert_index_period(self, insert, coerced_val, coerced_dtype):
        obj = pd.PeriodIndex(['2011-01', '2011-02', '2011-03', '2011-04'], freq='M')
        assert obj.dtype == 'period[M]'
        data = [pd.Period('2011-01', freq='M'), coerced_val, pd.Period('2011-02', freq='M'), pd.Period('2011-03', freq='M'), pd.Period('2011-04', freq='M')]
        if isinstance(insert, pd.Period):
            exp = pd.PeriodIndex(data, freq='M')
            self._assert_insert_conversion(obj, insert, exp, coerced_dtype)
            self._assert_insert_conversion(obj, str(insert), exp, coerced_dtype)
        else:
            result = obj.insert(0, insert)
            expected = obj.astype(object).insert(0, insert)
            tm.assert_index_equal(result, expected)
            if not isinstance(insert, pd.Timestamp):
                result = obj.insert(0, str(insert))
                expected = obj.astype(object).insert(0, str(insert))
                tm.assert_index_equal(result, expected)

    @pytest.mark.xfail(reason='Test not implemented')
    def test_insert_index_complex128(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_insert_index_bool(self):
        raise NotImplementedError