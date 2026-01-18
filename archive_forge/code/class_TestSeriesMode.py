from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
class TestSeriesMode:

    @pytest.mark.parametrize('dropna, expected', [(True, Series([], dtype=np.float64)), (False, Series([], dtype=np.float64))])
    def test_mode_empty(self, dropna, expected):
        s = Series([], dtype=np.float64)
        result = s.mode(dropna)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dropna, data, expected', [(True, [1, 1, 1, 2], [1]), (True, [1, 1, 1, 2, 3, 3, 3], [1, 3]), (False, [1, 1, 1, 2], [1]), (False, [1, 1, 1, 2, 3, 3, 3], [1, 3])])
    @pytest.mark.parametrize('dt', list(np.typecodes['AllInteger'] + np.typecodes['Float']))
    def test_mode_numerical(self, dropna, data, expected, dt):
        s = Series(data, dtype=dt)
        result = s.mode(dropna)
        expected = Series(expected, dtype=dt)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dropna, expected', [(True, [1.0]), (False, [1, np.nan])])
    def test_mode_numerical_nan(self, dropna, expected):
        s = Series([1, 1, 2, np.nan, np.nan])
        result = s.mode(dropna)
        expected = Series(expected)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dropna, expected1, expected2, expected3', [(True, ['b'], ['bar'], ['nan']), (False, ['b'], [np.nan], ['nan'])])
    def test_mode_str_obj(self, dropna, expected1, expected2, expected3):
        data = ['a'] * 2 + ['b'] * 3
        s = Series(data, dtype='c')
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype='c')
        tm.assert_series_equal(result, expected1)
        data = ['foo', 'bar', 'bar', np.nan, np.nan, np.nan]
        s = Series(data, dtype=object)
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype=None if expected2 == ['bar'] else object)
        tm.assert_series_equal(result, expected2)
        data = ['foo', 'bar', 'bar', np.nan, np.nan, np.nan]
        s = Series(data, dtype=object).astype(str)
        result = s.mode(dropna)
        expected3 = Series(expected3)
        tm.assert_series_equal(result, expected3)

    @pytest.mark.parametrize('dropna, expected1, expected2', [(True, ['foo'], ['foo']), (False, ['foo'], [np.nan])])
    def test_mode_mixeddtype(self, dropna, expected1, expected2):
        s = Series([1, 'foo', 'foo'])
        result = s.mode(dropna)
        expected = Series(expected1)
        tm.assert_series_equal(result, expected)
        s = Series([1, 'foo', 'foo', np.nan, np.nan, np.nan])
        result = s.mode(dropna)
        expected = Series(expected2, dtype=None if expected2 == ['foo'] else object)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dropna, expected1, expected2', [(True, ['1900-05-03', '2011-01-03', '2013-01-02'], ['2011-01-03', '2013-01-02']), (False, [np.nan], [np.nan, '2011-01-03', '2013-01-02'])])
    def test_mode_datetime(self, dropna, expected1, expected2):
        s = Series(['2011-01-03', '2013-01-02', '1900-05-03', 'nan', 'nan'], dtype='M8[ns]')
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype='M8[ns]')
        tm.assert_series_equal(result, expected1)
        s = Series(['2011-01-03', '2013-01-02', '1900-05-03', '2011-01-03', '2013-01-02', 'nan', 'nan'], dtype='M8[ns]')
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype='M8[ns]')
        tm.assert_series_equal(result, expected2)

    @pytest.mark.parametrize('dropna, expected1, expected2', [(True, ['-1 days', '0 days', '1 days'], ['2 min', '1 day']), (False, [np.nan], [np.nan, '2 min', '1 day'])])
    def test_mode_timedelta(self, dropna, expected1, expected2):
        s = Series(['1 days', '-1 days', '0 days', 'nan', 'nan'], dtype='timedelta64[ns]')
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype='timedelta64[ns]')
        tm.assert_series_equal(result, expected1)
        s = Series(['1 day', '1 day', '-1 day', '-1 day 2 min', '2 min', '2 min', 'nan', 'nan'], dtype='timedelta64[ns]')
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype='timedelta64[ns]')
        tm.assert_series_equal(result, expected2)

    @pytest.mark.parametrize('dropna, expected1, expected2, expected3', [(True, Categorical([1, 2], categories=[1, 2]), Categorical(['a'], categories=[1, 'a']), Categorical([3, 1], categories=[3, 2, 1], ordered=True)), (False, Categorical([np.nan], categories=[1, 2]), Categorical([np.nan, 'a'], categories=[1, 'a']), Categorical([np.nan, 3, 1], categories=[3, 2, 1], ordered=True))])
    def test_mode_category(self, dropna, expected1, expected2, expected3):
        s = Series(Categorical([1, 2, np.nan, np.nan]))
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype='category')
        tm.assert_series_equal(result, expected1)
        s = Series(Categorical([1, 'a', 'a', np.nan, np.nan]))
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype='category')
        tm.assert_series_equal(result, expected2)
        s = Series(Categorical([1, 1, 2, 3, 3, np.nan, np.nan], categories=[3, 2, 1], ordered=True))
        result = s.mode(dropna)
        expected3 = Series(expected3, dtype='category')
        tm.assert_series_equal(result, expected3)

    @pytest.mark.parametrize('dropna, expected1, expected2', [(True, [2 ** 63], [1, 2 ** 63]), (False, [2 ** 63], [1, 2 ** 63])])
    def test_mode_intoverflow(self, dropna, expected1, expected2):
        s = Series([1, 2 ** 63, 2 ** 63], dtype=np.uint64)
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype=np.uint64)
        tm.assert_series_equal(result, expected1)
        s = Series([1, 2 ** 63], dtype=np.uint64)
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype=np.uint64)
        tm.assert_series_equal(result, expected2)

    def test_mode_sortwarning(self):
        expected = Series(['foo', np.nan])
        s = Series([1, 'foo', 'foo', np.nan, np.nan])
        with tm.assert_produces_warning(UserWarning):
            result = s.mode(dropna=False)
            result = result.sort_values().reset_index(drop=True)
        tm.assert_series_equal(result, expected)

    def test_mode_boolean_with_na(self):
        ser = Series([True, False, True, pd.NA], dtype='boolean')
        result = ser.mode()
        expected = Series({0: True}, dtype='boolean')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('array,expected,dtype', [([0, 1j, 1, 1, 1 + 1j, 1 + 2j], Series([1], dtype=np.complex128), np.complex128), ([0, 1j, 1, 1, 1 + 1j, 1 + 2j], Series([1], dtype=np.complex64), np.complex64), ([1 + 1j, 2j, 1 + 1j], Series([1 + 1j], dtype=np.complex128), np.complex128)])
    def test_single_mode_value_complex(self, array, expected, dtype):
        result = Series(array, dtype=dtype).mode()
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('array,expected,dtype', [([0, 1j, 1, 1 + 1j, 1 + 2j], Series([0j, 1j, 1 + 0j, 1 + 1j, 1 + 2j], dtype=np.complex128), np.complex128), ([1 + 1j, 2j, 1 + 1j, 2j, 3], Series([2j, 1 + 1j], dtype=np.complex64), np.complex64)])
    def test_multimode_complex(self, array, expected, dtype):
        result = Series(array, dtype=dtype).mode()
        tm.assert_series_equal(result, expected)