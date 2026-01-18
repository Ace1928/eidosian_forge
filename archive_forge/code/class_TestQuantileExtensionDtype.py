import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestQuantileExtensionDtype:

    @pytest.fixture(params=[pytest.param(pd.IntervalIndex.from_breaks(range(10)), marks=pytest.mark.xfail(reason='raises when trying to add Intervals')), pd.period_range('2016-01-01', periods=9, freq='D'), pd.date_range('2016-01-01', periods=9, tz='US/Pacific'), pd.timedelta_range('1 Day', periods=9), pd.array(np.arange(9), dtype='Int64'), pd.array(np.arange(9), dtype='Float64')], ids=lambda x: str(x.dtype))
    def index(self, request):
        idx = request.param
        idx.name = 'A'
        return idx

    @pytest.fixture
    def obj(self, index, frame_or_series):
        obj = frame_or_series(index).copy()
        if frame_or_series is Series:
            obj.name = 'A'
        else:
            obj.columns = ['A']
        return obj

    def compute_quantile(self, obj, qs):
        if isinstance(obj, Series):
            result = obj.quantile(qs)
        else:
            result = obj.quantile(qs, numeric_only=False)
        return result

    def test_quantile_ea(self, request, obj, index):
        indexer = np.arange(len(index), dtype=np.intp)
        np.random.default_rng(2).shuffle(indexer)
        obj = obj.iloc[indexer]
        qs = [0.5, 0, 1]
        result = self.compute_quantile(obj, qs)
        exp_dtype = index.dtype
        if index.dtype == 'Int64':
            exp_dtype = 'Float64'
        expected = Series([index[4], index[0], index[-1]], dtype=exp_dtype, index=qs, name='A')
        expected = type(obj)(expected)
        tm.assert_equal(result, expected)

    def test_quantile_ea_with_na(self, obj, index):
        obj.iloc[0] = index._na_value
        obj.iloc[-1] = index._na_value
        indexer = np.arange(len(index), dtype=np.intp)
        np.random.default_rng(2).shuffle(indexer)
        obj = obj.iloc[indexer]
        qs = [0.5, 0, 1]
        result = self.compute_quantile(obj, qs)
        expected = Series([index[4], index[1], index[-2]], dtype=index.dtype, index=qs, name='A')
        expected = type(obj)(expected)
        tm.assert_equal(result, expected)

    def test_quantile_ea_all_na(self, request, obj, index):
        obj.iloc[:] = index._na_value
        assert np.all(obj.dtypes == index.dtype)
        indexer = np.arange(len(index), dtype=np.intp)
        np.random.default_rng(2).shuffle(indexer)
        obj = obj.iloc[indexer]
        qs = [0.5, 0, 1]
        result = self.compute_quantile(obj, qs)
        expected = index.take([-1, -1, -1], allow_fill=True, fill_value=index._na_value)
        expected = Series(expected, index=qs, name='A')
        expected = type(obj)(expected)
        tm.assert_equal(result, expected)

    def test_quantile_ea_scalar(self, request, obj, index):
        indexer = np.arange(len(index), dtype=np.intp)
        np.random.default_rng(2).shuffle(indexer)
        obj = obj.iloc[indexer]
        qs = 0.5
        result = self.compute_quantile(obj, qs)
        exp_dtype = index.dtype
        if index.dtype == 'Int64':
            exp_dtype = 'Float64'
        expected = Series({'A': index[4]}, dtype=exp_dtype, name=0.5)
        if isinstance(obj, Series):
            expected = expected['A']
            assert result == expected
        else:
            tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dtype, expected_data, expected_index, axis', [['float64', [], [], 1], ['int64', [], [], 1], ['float64', [np.nan, np.nan], ['a', 'b'], 0], ['int64', [np.nan, np.nan], ['a', 'b'], 0]])
    def test_empty_numeric(self, dtype, expected_data, expected_index, axis):
        df = DataFrame(columns=['a', 'b'], dtype=dtype)
        result = df.quantile(0.5, axis=axis)
        expected = Series(expected_data, name=0.5, index=Index(expected_index), dtype='float64')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dtype, expected_data, expected_index, axis, expected_dtype', [['datetime64[ns]', [], [], 1, 'datetime64[ns]'], ['datetime64[ns]', [pd.NaT, pd.NaT], ['a', 'b'], 0, 'datetime64[ns]']])
    def test_empty_datelike(self, dtype, expected_data, expected_index, axis, expected_dtype):
        df = DataFrame(columns=['a', 'b'], dtype=dtype)
        result = df.quantile(0.5, axis=axis, numeric_only=False)
        expected = Series(expected_data, name=0.5, index=Index(expected_index), dtype=expected_dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('expected_data, expected_index, axis', [[[np.nan, np.nan], range(2), 1], [[], [], 0]])
    def test_datelike_numeric_only(self, expected_data, expected_index, axis):
        df = DataFrame({'a': pd.to_datetime(['2010', '2011']), 'b': [0, 5], 'c': pd.to_datetime(['2011', '2012'])})
        result = df[['a', 'c']].quantile(0.5, axis=axis, numeric_only=True)
        expected = Series(expected_data, name=0.5, index=Index(expected_index), dtype=np.float64)
        tm.assert_series_equal(result, expected)