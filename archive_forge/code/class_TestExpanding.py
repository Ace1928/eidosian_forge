import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
class TestExpanding:

    @pytest.fixture
    def frame(self):
        return DataFrame({'A': [1] * 20 + [2] * 12 + [3] * 8, 'B': np.arange(40)})

    @pytest.mark.parametrize('f', ['sum', 'mean', 'min', 'max', 'count', 'kurt', 'skew'])
    def test_expanding(self, f, frame):
        g = frame.groupby('A', group_keys=False)
        r = g.expanding()
        result = getattr(r, f)()
        msg = 'DataFrameGroupBy.apply operated on the grouping columns'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            expected = g.apply(lambda x: getattr(x.expanding(), f)())
        expected = expected.drop('A', axis=1)
        expected_index = MultiIndex.from_arrays([frame['A'], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('f', ['std', 'var'])
    def test_expanding_ddof(self, f, frame):
        g = frame.groupby('A', group_keys=False)
        r = g.expanding()
        result = getattr(r, f)(ddof=0)
        msg = 'DataFrameGroupBy.apply operated on the grouping columns'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            expected = g.apply(lambda x: getattr(x.expanding(), f)(ddof=0))
        expected = expected.drop('A', axis=1)
        expected_index = MultiIndex.from_arrays([frame['A'], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('interpolation', ['linear', 'lower', 'higher', 'midpoint', 'nearest'])
    def test_expanding_quantile(self, interpolation, frame):
        g = frame.groupby('A', group_keys=False)
        r = g.expanding()
        result = r.quantile(0.4, interpolation=interpolation)
        msg = 'DataFrameGroupBy.apply operated on the grouping columns'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            expected = g.apply(lambda x: x.expanding().quantile(0.4, interpolation=interpolation))
        expected = expected.drop('A', axis=1)
        expected_index = MultiIndex.from_arrays([frame['A'], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('f', ['corr', 'cov'])
    def test_expanding_corr_cov(self, f, frame):
        g = frame.groupby('A')
        r = g.expanding()
        result = getattr(r, f)(frame)

        def func_0(x):
            return getattr(x.expanding(), f)(frame)
        msg = 'DataFrameGroupBy.apply operated on the grouping columns'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            expected = g.apply(func_0)
        null_idx = list(range(20, 61)) + list(range(72, 113))
        expected.iloc[null_idx, 1] = np.nan
        expected['A'] = np.nan
        tm.assert_frame_equal(result, expected)
        result = getattr(r.B, f)(pairwise=True)

        def func_1(x):
            return getattr(x.B.expanding(), f)(pairwise=True)
        msg = 'DataFrameGroupBy.apply operated on the grouping columns'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            expected = g.apply(func_1)
        tm.assert_series_equal(result, expected)

    def test_expanding_apply(self, raw, frame):
        g = frame.groupby('A', group_keys=False)
        r = g.expanding()
        result = r.apply(lambda x: x.sum(), raw=raw)
        msg = 'DataFrameGroupBy.apply operated on the grouping columns'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            expected = g.apply(lambda x: x.expanding().apply(lambda y: y.sum(), raw=raw))
        expected = expected.drop('A', axis=1)
        expected_index = MultiIndex.from_arrays([frame['A'], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)