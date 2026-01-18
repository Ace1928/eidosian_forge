import calendar
import datetime
import decimal
import json
import locale
import math
import re
import time
import dateutil
import numpy as np
import pytest
import pytz
import pandas._libs.json as ujson
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
class TestPandasJSONTests:

    def test_dataframe(self, orient):
        dtype = np.int64
        df = DataFrame([[1, 2, 3], [4, 5, 6]], index=['a', 'b'], columns=['x', 'y', 'z'], dtype=dtype)
        encode_kwargs = {} if orient is None else {'orient': orient}
        assert (df.dtypes == dtype).all()
        output = ujson.ujson_loads(ujson.ujson_dumps(df, **encode_kwargs))
        assert (df.dtypes == dtype).all()
        if orient == 'split':
            dec = _clean_dict(output)
            output = DataFrame(**dec)
        else:
            output = DataFrame(output)
        if orient == 'values':
            df.columns = [0, 1, 2]
            df.index = [0, 1]
        elif orient == 'records':
            df.index = [0, 1]
        elif orient == 'index':
            df = df.transpose()
        assert (df.dtypes == dtype).all()
        tm.assert_frame_equal(output, df)

    def test_dataframe_nested(self, orient):
        df = DataFrame([[1, 2, 3], [4, 5, 6]], index=['a', 'b'], columns=['x', 'y', 'z'])
        nested = {'df1': df, 'df2': df.copy()}
        kwargs = {} if orient is None else {'orient': orient}
        exp = {'df1': ujson.ujson_loads(ujson.ujson_dumps(df, **kwargs)), 'df2': ujson.ujson_loads(ujson.ujson_dumps(df, **kwargs))}
        assert ujson.ujson_loads(ujson.ujson_dumps(nested, **kwargs)) == exp

    def test_series(self, orient):
        dtype = np.int64
        s = Series([10, 20, 30, 40, 50, 60], name='series', index=[6, 7, 8, 9, 10, 15], dtype=dtype).sort_values()
        assert s.dtype == dtype
        encode_kwargs = {} if orient is None else {'orient': orient}
        output = ujson.ujson_loads(ujson.ujson_dumps(s, **encode_kwargs))
        assert s.dtype == dtype
        if orient == 'split':
            dec = _clean_dict(output)
            output = Series(**dec)
        else:
            output = Series(output)
        if orient in (None, 'index'):
            s.name = None
            output = output.sort_values()
            s.index = ['6', '7', '8', '9', '10', '15']
        elif orient in ('records', 'values'):
            s.name = None
            s.index = [0, 1, 2, 3, 4, 5]
        assert s.dtype == dtype
        tm.assert_series_equal(output, s)

    def test_series_nested(self, orient):
        s = Series([10, 20, 30, 40, 50, 60], name='series', index=[6, 7, 8, 9, 10, 15]).sort_values()
        nested = {'s1': s, 's2': s.copy()}
        kwargs = {} if orient is None else {'orient': orient}
        exp = {'s1': ujson.ujson_loads(ujson.ujson_dumps(s, **kwargs)), 's2': ujson.ujson_loads(ujson.ujson_dumps(s, **kwargs))}
        assert ujson.ujson_loads(ujson.ujson_dumps(nested, **kwargs)) == exp

    def test_index(self):
        i = Index([23, 45, 18, 98, 43, 11], name='index')
        output = Index(ujson.ujson_loads(ujson.ujson_dumps(i)), name='index')
        tm.assert_index_equal(i, output)
        dec = _clean_dict(ujson.ujson_loads(ujson.ujson_dumps(i, orient='split')))
        output = Index(**dec)
        tm.assert_index_equal(i, output)
        assert i.name == output.name
        tm.assert_index_equal(i, output)
        assert i.name == output.name
        output = Index(ujson.ujson_loads(ujson.ujson_dumps(i, orient='values')), name='index')
        tm.assert_index_equal(i, output)
        output = Index(ujson.ujson_loads(ujson.ujson_dumps(i, orient='records')), name='index')
        tm.assert_index_equal(i, output)
        output = Index(ujson.ujson_loads(ujson.ujson_dumps(i, orient='index')), name='index')
        tm.assert_index_equal(i, output)

    def test_datetime_index(self):
        date_unit = 'ns'
        rng = DatetimeIndex(list(date_range('1/1/2000', periods=20)), freq=None)
        encoded = ujson.ujson_dumps(rng, date_unit=date_unit)
        decoded = DatetimeIndex(np.array(ujson.ujson_loads(encoded)))
        tm.assert_index_equal(rng, decoded)
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
        decoded = Series(ujson.ujson_loads(ujson.ujson_dumps(ts, date_unit=date_unit)))
        idx_values = decoded.index.values.astype(np.int64)
        decoded.index = DatetimeIndex(idx_values)
        tm.assert_series_equal(ts, decoded)

    @pytest.mark.parametrize('invalid_arr', ['[31337,]', '[,31337]', '[]]', '[,]'])
    def test_decode_invalid_array(self, invalid_arr):
        msg = 'Expected object or value|Trailing data|Unexpected character found when decoding array value'
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(invalid_arr)

    @pytest.mark.parametrize('arr', [[], [31337]])
    def test_decode_array(self, arr):
        assert arr == ujson.ujson_loads(str(arr))

    @pytest.mark.parametrize('extreme_num', [9223372036854775807, -9223372036854775808])
    def test_decode_extreme_numbers(self, extreme_num):
        assert extreme_num == ujson.ujson_loads(str(extreme_num))

    @pytest.mark.parametrize('too_extreme_num', [f'{2 ** 64}', f'{-2 ** 63 - 1}'])
    def test_decode_too_extreme_numbers(self, too_extreme_num):
        with pytest.raises(ValueError, match='Value is too big|Value is too small'):
            ujson.ujson_loads(too_extreme_num)

    def test_decode_with_trailing_whitespaces(self):
        assert {} == ujson.ujson_loads('{}\n\t ')

    def test_decode_with_trailing_non_whitespaces(self):
        with pytest.raises(ValueError, match='Trailing data'):
            ujson.ujson_loads('{}\n\t a')

    @pytest.mark.parametrize('value', [f'{2 ** 64}', f'{-2 ** 63 - 1}'])
    def test_decode_array_with_big_int(self, value):
        with pytest.raises(ValueError, match='Value is too big|Value is too small'):
            ujson.ujson_loads(value)

    @pytest.mark.parametrize('float_number', [1.1234567893, 1.234567893, 1.34567893, 1.4567893, 1.567893, 1.67893, 1.7893, 1.893, 1.3])
    @pytest.mark.parametrize('sign', [-1, 1])
    def test_decode_floating_point(self, sign, float_number):
        float_number *= sign
        tm.assert_almost_equal(float_number, ujson.ujson_loads(str(float_number)), rtol=1e-15)

    def test_encode_big_set(self):
        s = set()
        for x in range(100000):
            s.add(x)
        ujson.ujson_dumps(s)

    def test_encode_empty_set(self):
        assert '[]' == ujson.ujson_dumps(set())

    def test_encode_set(self):
        s = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        enc = ujson.ujson_dumps(s)
        dec = ujson.ujson_loads(enc)
        for v in dec:
            assert v in s

    @pytest.mark.parametrize('td', [Timedelta(days=366), Timedelta(days=-1), Timedelta(hours=13, minutes=5, seconds=5), Timedelta(hours=13, minutes=20, seconds=30), Timedelta(days=-1, nanoseconds=5), Timedelta(nanoseconds=1), Timedelta(microseconds=1, nanoseconds=1), Timedelta(milliseconds=1, microseconds=1, nanoseconds=1), Timedelta(milliseconds=999, microseconds=999, nanoseconds=999)])
    def test_encode_timedelta_iso(self, td):
        result = ujson.ujson_dumps(td, iso_dates=True)
        expected = f'"{td.isoformat()}"'
        assert result == expected

    def test_encode_periodindex(self):
        p = PeriodIndex(['2022-04-06', '2022-04-07'], freq='D')
        df = DataFrame(index=p)
        assert df.to_json() == '{}'