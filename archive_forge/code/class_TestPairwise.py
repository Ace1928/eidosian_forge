import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
class TestPairwise:

    @pytest.mark.parametrize('f', [lambda x: x.cov(), lambda x: x.corr()])
    def test_no_flex(self, pairwise_frames, pairwise_target_frame, f):
        result = f(pairwise_frames)
        tm.assert_index_equal(result.index, pairwise_frames.columns)
        tm.assert_index_equal(result.columns, pairwise_frames.columns)
        expected = f(pairwise_target_frame)
        result = result.dropna().values
        expected = expected.dropna().values
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize('f', [lambda x: x.expanding().cov(pairwise=True), lambda x: x.expanding().corr(pairwise=True), lambda x: x.rolling(window=3).cov(pairwise=True), lambda x: x.rolling(window=3).corr(pairwise=True), lambda x: x.ewm(com=3).cov(pairwise=True), lambda x: x.ewm(com=3).corr(pairwise=True)])
    def test_pairwise_with_self(self, pairwise_frames, pairwise_target_frame, f):
        result = f(pairwise_frames)
        tm.assert_index_equal(result.index.levels[0], pairwise_frames.index, check_names=False)
        tm.assert_index_equal(safe_sort(result.index.levels[1]), safe_sort(pairwise_frames.columns.unique()))
        tm.assert_index_equal(result.columns, pairwise_frames.columns)
        expected = f(pairwise_target_frame)
        result = result.dropna().values
        expected = expected.dropna().values
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize('f', [lambda x: x.expanding().cov(pairwise=False), lambda x: x.expanding().corr(pairwise=False), lambda x: x.rolling(window=3).cov(pairwise=False), lambda x: x.rolling(window=3).corr(pairwise=False), lambda x: x.ewm(com=3).cov(pairwise=False), lambda x: x.ewm(com=3).corr(pairwise=False)])
    def test_no_pairwise_with_self(self, pairwise_frames, pairwise_target_frame, f):
        result = f(pairwise_frames)
        tm.assert_index_equal(result.index, pairwise_frames.index)
        tm.assert_index_equal(result.columns, pairwise_frames.columns)
        expected = f(pairwise_target_frame)
        result = result.dropna().values
        expected = expected.dropna().values
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize('f', [lambda x, y: x.expanding().cov(y, pairwise=True), lambda x, y: x.expanding().corr(y, pairwise=True), lambda x, y: x.rolling(window=3).cov(y, pairwise=True), pytest.param(lambda x, y: x.rolling(window=3).corr(y, pairwise=True), marks=pytest.mark.xfail(not IS64, reason='Precision issues on 32 bit', strict=False)), lambda x, y: x.ewm(com=3).cov(y, pairwise=True), lambda x, y: x.ewm(com=3).corr(y, pairwise=True)])
    def test_pairwise_with_other(self, pairwise_frames, pairwise_target_frame, pairwise_other_frame, f):
        result = f(pairwise_frames, pairwise_other_frame)
        tm.assert_index_equal(result.index.levels[0], pairwise_frames.index, check_names=False)
        tm.assert_index_equal(safe_sort(result.index.levels[1]), safe_sort(pairwise_other_frame.columns.unique()))
        expected = f(pairwise_target_frame, pairwise_other_frame)
        result = result.dropna().values
        expected = expected.dropna().values
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    @pytest.mark.filterwarnings('ignore:RuntimeWarning')
    @pytest.mark.parametrize('f', [lambda x, y: x.expanding().cov(y, pairwise=False), lambda x, y: x.expanding().corr(y, pairwise=False), lambda x, y: x.rolling(window=3).cov(y, pairwise=False), lambda x, y: x.rolling(window=3).corr(y, pairwise=False), lambda x, y: x.ewm(com=3).cov(y, pairwise=False), lambda x, y: x.ewm(com=3).corr(y, pairwise=False)])
    def test_no_pairwise_with_other(self, pairwise_frames, pairwise_other_frame, f):
        result = f(pairwise_frames, pairwise_other_frame) if pairwise_frames.columns.is_unique else None
        if result is not None:
            expected_index = pairwise_frames.index.union(pairwise_other_frame.index)
            expected_columns = pairwise_frames.columns.union(pairwise_other_frame.columns)
            tm.assert_index_equal(result.index, expected_index)
            tm.assert_index_equal(result.columns, expected_columns)
        else:
            with pytest.raises(ValueError, match="'arg1' columns are not unique"):
                f(pairwise_frames, pairwise_other_frame)
            with pytest.raises(ValueError, match="'arg2' columns are not unique"):
                f(pairwise_other_frame, pairwise_frames)

    @pytest.mark.parametrize('f', [lambda x, y: x.expanding().cov(y), lambda x, y: x.expanding().corr(y), lambda x, y: x.rolling(window=3).cov(y), lambda x, y: x.rolling(window=3).corr(y), lambda x, y: x.ewm(com=3).cov(y), lambda x, y: x.ewm(com=3).corr(y)])
    def test_pairwise_with_series(self, pairwise_frames, pairwise_target_frame, f):
        result = f(pairwise_frames, Series([1, 1, 3, 8]))
        tm.assert_index_equal(result.index, pairwise_frames.index)
        tm.assert_index_equal(result.columns, pairwise_frames.columns)
        expected = f(pairwise_target_frame, Series([1, 1, 3, 8]))
        result = result.dropna().values
        expected = expected.dropna().values
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)
        result = f(Series([1, 1, 3, 8]), pairwise_frames)
        tm.assert_index_equal(result.index, pairwise_frames.index)
        tm.assert_index_equal(result.columns, pairwise_frames.columns)
        expected = f(Series([1, 1, 3, 8]), pairwise_target_frame)
        result = result.dropna().values
        expected = expected.dropna().values
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    def test_corr_freq_memory_error(self):
        s = Series(range(5), index=date_range('2020', periods=5))
        result = s.rolling('12h').corr(s)
        expected = Series([np.nan] * 5, index=date_range('2020', periods=5))
        tm.assert_series_equal(result, expected)

    def test_cov_mulittindex(self):
        columns = MultiIndex.from_product([list('ab'), list('xy'), list('AB')])
        index = range(3)
        df = DataFrame(np.arange(24).reshape(3, 8), index=index, columns=columns)
        result = df.ewm(alpha=0.1).cov()
        index = MultiIndex.from_product([range(3), list('ab'), list('xy'), list('AB')])
        columns = MultiIndex.from_product([list('ab'), list('xy'), list('AB')])
        expected = DataFrame(np.vstack((np.full((8, 8), np.nan), np.full((8, 8), 32.0), np.full((8, 8), 63.881919))), index=index, columns=columns)
        tm.assert_frame_equal(result, expected)

    def test_multindex_columns_pairwise_func(self):
        columns = MultiIndex.from_arrays([['M', 'N'], ['P', 'Q']], names=['a', 'b'])
        df = DataFrame(np.ones((5, 2)), columns=columns)
        result = df.rolling(3).corr()
        expected = DataFrame(np.nan, index=MultiIndex.from_arrays([np.repeat(np.arange(5, dtype=np.int64), 2), ['M', 'N'] * 5, ['P', 'Q'] * 5], names=[None, 'a', 'b']), columns=columns)
        tm.assert_frame_equal(result, expected)