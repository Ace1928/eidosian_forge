import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestDataFrameCumulativeOps:

    def test_cumulative_ops_smoke(self):
        df = DataFrame({'A': np.arange(20)}, index=np.arange(20))
        df.cummax()
        df.cummin()
        df.cumsum()
        dm = DataFrame(np.arange(20).reshape(4, 5), index=range(4), columns=range(5))
        dm.cumsum()

    def test_cumprod_smoke(self, datetime_frame):
        datetime_frame.iloc[5:10, 0] = np.nan
        datetime_frame.iloc[10:15, 1] = np.nan
        datetime_frame.iloc[15:, 2] = np.nan
        df = datetime_frame.fillna(0).astype(int)
        df.cumprod(0)
        df.cumprod(1)
        df = datetime_frame.fillna(0).astype(np.int32)
        df.cumprod(0)
        df.cumprod(1)

    @pytest.mark.parametrize('method', ['cumsum', 'cumprod', 'cummin', 'cummax'])
    def test_cumulative_ops_match_series_apply(self, datetime_frame, method):
        datetime_frame.iloc[5:10, 0] = np.nan
        datetime_frame.iloc[10:15, 1] = np.nan
        datetime_frame.iloc[15:, 2] = np.nan
        result = getattr(datetime_frame, method)()
        expected = datetime_frame.apply(getattr(Series, method))
        tm.assert_frame_equal(result, expected)
        result = getattr(datetime_frame, method)(axis=1)
        expected = datetime_frame.apply(getattr(Series, method), axis=1)
        tm.assert_frame_equal(result, expected)
        assert np.shape(result) == np.shape(datetime_frame)

    def test_cumsum_preserve_dtypes(self):
        df = DataFrame({'A': [1, 2, 3], 'B': [1, 2, 3.0], 'C': [True, False, False]})
        result = df.cumsum()
        expected = DataFrame({'A': Series([1, 3, 6], dtype=np.int64), 'B': Series([1, 3, 6], dtype=np.float64), 'C': df['C'].cumsum()})
        tm.assert_frame_equal(result, expected)