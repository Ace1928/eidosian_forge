from datetime import (
from io import StringIO
import re
import sys
from textwrap import dedent
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
class TestSeriesToString:

    def test_to_string_without_index(self):
        ser = Series([1, 2, 3, 4])
        result = ser.to_string(index=False)
        expected = '\n'.join(['1', '2', '3', '4'])
        assert result == expected

    def test_to_string_name(self):
        ser = Series(range(100), dtype='int64')
        ser.name = 'myser'
        res = ser.to_string(max_rows=2, name=True)
        exp = '0      0\n      ..\n99    99\nName: myser'
        assert res == exp
        res = ser.to_string(max_rows=2, name=False)
        exp = '0      0\n      ..\n99    99'
        assert res == exp

    def test_to_string_dtype(self):
        ser = Series(range(100), dtype='int64')
        res = ser.to_string(max_rows=2, dtype=True)
        exp = '0      0\n      ..\n99    99\ndtype: int64'
        assert res == exp
        res = ser.to_string(max_rows=2, dtype=False)
        exp = '0      0\n      ..\n99    99'
        assert res == exp

    def test_to_string_length(self):
        ser = Series(range(100), dtype='int64')
        res = ser.to_string(max_rows=2, length=True)
        exp = '0      0\n      ..\n99    99\nLength: 100'
        assert res == exp

    def test_to_string_na_rep(self):
        ser = Series(index=range(100), dtype=np.float64)
        res = ser.to_string(na_rep='foo', max_rows=2)
        exp = '0    foo\n      ..\n99   foo'
        assert res == exp

    def test_to_string_float_format(self):
        ser = Series(range(10), dtype='float64')
        res = ser.to_string(float_format=lambda x: f'{x:2.1f}', max_rows=2)
        exp = '0   0.0\n     ..\n9   9.0'
        assert res == exp

    def test_to_string_header(self):
        ser = Series(range(10), dtype='int64')
        ser.index.name = 'foo'
        res = ser.to_string(header=True, max_rows=2)
        exp = 'foo\n0    0\n    ..\n9    9'
        assert res == exp
        res = ser.to_string(header=False, max_rows=2)
        exp = '0    0\n    ..\n9    9'
        assert res == exp

    def test_to_string_empty_col(self):
        ser = Series(['', 'Hello', 'World', '', '', 'Mooooo', '', ''])
        res = ser.to_string(index=False)
        exp = '      \n Hello\n World\n      \n      \nMooooo\n      \n      '
        assert re.match(exp, res)

    def test_to_string_timedelta64(self):
        Series(np.array([1100, 20], dtype='timedelta64[ns]')).to_string()
        ser = Series(date_range('2012-1-1', periods=3, freq='D'))
        y = ser - ser.shift(1)
        result = y.to_string()
        assert '1 days' in result
        assert '00:00:00' not in result
        assert 'NaT' in result
        o = Series([datetime(2012, 1, 1, microsecond=150)] * 3)
        y = ser - o
        result = y.to_string()
        assert '-1 days +23:59:59.999850' in result
        o = Series([datetime(2012, 1, 1, 1)] * 3)
        y = ser - o
        result = y.to_string()
        assert '-1 days +23:00:00' in result
        assert '1 days 23:00:00' in result
        o = Series([datetime(2012, 1, 1, 1, 1)] * 3)
        y = ser - o
        result = y.to_string()
        assert '-1 days +22:59:00' in result
        assert '1 days 22:59:00' in result
        o = Series([datetime(2012, 1, 1, 1, 1, microsecond=150)] * 3)
        y = ser - o
        result = y.to_string()
        assert '-1 days +22:58:59.999850' in result
        assert '0 days 22:58:59.999850' in result
        td = timedelta(minutes=5, seconds=3)
        s2 = Series(date_range('2012-1-1', periods=3, freq='D')) + td
        y = ser - s2
        result = y.to_string()
        assert '-1 days +23:54:57' in result
        td = timedelta(microseconds=550)
        s2 = Series(date_range('2012-1-1', periods=3, freq='D')) + td
        y = ser - td
        result = y.to_string()
        assert '2012-01-01 23:59:59.999450' in result
        td = Series(timedelta_range('1 days', periods=3))
        result = td.to_string()
        assert result == '0   1 days\n1   2 days\n2   3 days'

    def test_to_string(self):
        ts = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10, freq='B'))
        buf = StringIO()
        s = ts.to_string()
        retval = ts.to_string(buf=buf)
        assert retval is None
        assert buf.getvalue().strip() == s
        format = '%.4f'.__mod__
        result = ts.to_string(float_format=format)
        result = [x.split()[1] for x in result.split('\n')[:-1]]
        expected = [format(x) for x in ts]
        assert result == expected
        result = ts[:0].to_string()
        assert result == 'Series([], Freq: B)'
        result = ts[:0].to_string(length=0)
        assert result == 'Series([], Freq: B)'
        cp = ts.copy()
        cp.name = 'foo'
        result = cp.to_string(length=True, name=True, dtype=True)
        last_line = result.split('\n')[-1].strip()
        assert last_line == f'Freq: B, Name: foo, Length: {len(cp)}, dtype: float64'

    @pytest.mark.parametrize('input_array, expected', [('a', 'a'), (['a', 'b'], 'a\nb'), ([1, 'a'], '1\na'), (1, '1'), ([0, -1], ' 0\n-1'), (1.0, '1.0'), ([' a', ' b'], ' a\n b'), (['.1', '1'], '.1\n 1'), (['10', '-10'], ' 10\n-10')])
    def test_format_remove_leading_space_series(self, input_array, expected):
        ser = Series(input_array)
        result = ser.to_string(index=False)
        assert result == expected

    def test_to_string_complex_number_trims_zeros(self):
        ser = Series([1.0 + 1j, 1.0 + 1j, 1.05 + 1j])
        result = ser.to_string()
        expected = dedent('            0    1.00+1.00j\n            1    1.00+1.00j\n            2    1.05+1.00j')
        assert result == expected

    def test_nullable_float_to_string(self, float_ea_dtype):
        dtype = float_ea_dtype
        ser = Series([0.0, 1.0, None], dtype=dtype)
        result = ser.to_string()
        expected = dedent('            0     0.0\n            1     1.0\n            2    <NA>')
        assert result == expected

    def test_nullable_int_to_string(self, any_int_ea_dtype):
        dtype = any_int_ea_dtype
        ser = Series([0, 1, None], dtype=dtype)
        result = ser.to_string()
        expected = dedent('            0       0\n            1       1\n            2    <NA>')
        assert result == expected

    def test_to_string_mixed(self):
        ser = Series(['foo', np.nan, -1.23, 4.56])
        result = ser.to_string()
        expected = ''.join(['0     foo\n', '1     NaN\n', '2   -1.23\n', '3    4.56'])
        assert result == expected
        ser = Series(['foo', np.nan, 'bar', 'baz'])
        result = ser.to_string()
        expected = ''.join(['0    foo\n', '1    NaN\n', '2    bar\n', '3    baz'])
        assert result == expected
        ser = Series(['foo', 5, 'bar', 'baz'])
        result = ser.to_string()
        expected = ''.join(['0    foo\n', '1      5\n', '2    bar\n', '3    baz'])
        assert result == expected

    def test_to_string_float_na_spacing(self):
        ser = Series([0.0, 1.5678, 2.0, -3.0, 4.0])
        ser[::2] = np.nan
        result = ser.to_string()
        expected = '0       NaN\n1    1.5678\n2       NaN\n3   -3.0000\n4       NaN'
        assert result == expected

    def test_to_string_with_datetimeindex(self):
        index = date_range('20130102', periods=6)
        ser = Series(1, index=index)
        result = ser.to_string()
        assert '2013-01-02' in result
        s2 = Series(2, index=[Timestamp('20130111'), NaT])
        ser = concat([s2, ser])
        result = ser.to_string()
        assert 'NaT' in result
        result = str(s2.index)
        assert 'NaT' in result