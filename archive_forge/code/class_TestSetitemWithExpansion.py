from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import IndexingError
from pandas.core.dtypes.common import is_list_like
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
class TestSetitemWithExpansion:

    def test_setitem_empty_series(self):
        key = Timestamp('2012-01-01')
        series = Series(dtype=object)
        series[key] = 47
        expected = Series(47, [key])
        tm.assert_series_equal(series, expected)

    def test_setitem_empty_series_datetimeindex_preserves_freq(self):
        dti = DatetimeIndex([], freq='D', dtype='M8[ns]')
        series = Series([], index=dti, dtype=object)
        key = Timestamp('2012-01-01')
        series[key] = 47
        expected = Series(47, DatetimeIndex([key], freq='D').as_unit('ns'))
        tm.assert_series_equal(series, expected)
        assert series.index.freq == expected.index.freq

    def test_setitem_empty_series_timestamp_preserves_dtype(self):
        timestamp = Timestamp(1412526600000000000)
        series = Series([timestamp], index=['timestamp'], dtype=object)
        expected = series['timestamp']
        series = Series([], dtype=object)
        series['anything'] = 300.0
        series['timestamp'] = timestamp
        result = series['timestamp']
        assert result == expected

    @pytest.mark.parametrize('td', [Timedelta('9 days'), Timedelta('9 days').to_timedelta64(), Timedelta('9 days').to_pytimedelta()])
    def test_append_timedelta_does_not_cast(self, td, using_infer_string, request):
        if using_infer_string and (not isinstance(td, Timedelta)):
            request.applymarker(pytest.mark.xfail(reason='inferred as string'))
        expected = Series(['x', td], index=[0, 'td'], dtype=object)
        ser = Series(['x'])
        ser['td'] = td
        tm.assert_series_equal(ser, expected)
        assert isinstance(ser['td'], Timedelta)
        ser = Series(['x'])
        ser.loc['td'] = Timedelta('9 days')
        tm.assert_series_equal(ser, expected)
        assert isinstance(ser['td'], Timedelta)

    def test_setitem_with_expansion_type_promotion(self):
        ser = Series(dtype=object)
        ser['a'] = Timestamp('2016-01-01')
        ser['b'] = 3.0
        ser['c'] = 'foo'
        expected = Series([Timestamp('2016-01-01'), 3.0, 'foo'], index=['a', 'b', 'c'])
        tm.assert_series_equal(ser, expected)

    def test_setitem_not_contained(self, string_series):
        ser = string_series.copy()
        assert 'foobar' not in ser.index
        ser['foobar'] = 1
        app = Series([1], index=['foobar'], name='series')
        expected = concat([string_series, app])
        tm.assert_series_equal(ser, expected)

    def test_setitem_keep_precision(self, any_numeric_ea_dtype):
        ser = Series([1, 2], dtype=any_numeric_ea_dtype)
        ser[2] = 10
        expected = Series([1, 2, 10], dtype=any_numeric_ea_dtype)
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize('na, target_na, dtype, target_dtype, indexer, warn', [(NA, NA, 'Int64', 'Int64', 1, None), (NA, NA, 'Int64', 'Int64', 2, None), (NA, np.nan, 'int64', 'float64', 1, None), (NA, np.nan, 'int64', 'float64', 2, None), (NaT, NaT, 'int64', 'object', 1, FutureWarning), (NaT, NaT, 'int64', 'object', 2, None), (np.nan, NA, 'Int64', 'Int64', 1, None), (np.nan, NA, 'Int64', 'Int64', 2, None), (np.nan, NA, 'Float64', 'Float64', 1, None), (np.nan, NA, 'Float64', 'Float64', 2, None), (np.nan, np.nan, 'int64', 'float64', 1, None), (np.nan, np.nan, 'int64', 'float64', 2, None)])
    def test_setitem_enlarge_with_na(self, na, target_na, dtype, target_dtype, indexer, warn):
        ser = Series([1, 2], dtype=dtype)
        with tm.assert_produces_warning(warn, match='incompatible dtype'):
            ser[indexer] = na
        expected_values = [1, target_na] if indexer == 1 else [1, 2, target_na]
        expected = Series(expected_values, dtype=target_dtype)
        tm.assert_series_equal(ser, expected)

    def test_setitem_enlargement_object_none(self, nulls_fixture, using_infer_string):
        ser = Series(['a', 'b'])
        ser[3] = nulls_fixture
        dtype = 'string[pyarrow_numpy]' if using_infer_string and (not isinstance(nulls_fixture, Decimal)) else object
        expected = Series(['a', 'b', nulls_fixture], index=[0, 1, 3], dtype=dtype)
        tm.assert_series_equal(ser, expected)
        if using_infer_string:
            ser[3] is np.nan
        else:
            assert ser[3] is nulls_fixture