import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestDataFrameCov:

    def test_cov(self, float_frame, float_string_frame):
        expected = float_frame.cov()
        result = float_frame.cov(min_periods=len(float_frame))
        tm.assert_frame_equal(expected, result)
        result = float_frame.cov(min_periods=len(float_frame) + 1)
        assert isna(result.values).all()
        frame = float_frame.copy()
        frame.iloc[:5, frame.columns.get_loc('A')] = np.nan
        frame.iloc[5:10, frame.columns.get_loc('B')] = np.nan
        result = frame.cov(min_periods=len(frame) - 8)
        expected = frame.cov()
        expected.loc['A', 'B'] = np.nan
        expected.loc['B', 'A'] = np.nan
        tm.assert_frame_equal(result, expected)
        result = frame.cov()
        expected = frame['A'].cov(frame['C'])
        tm.assert_almost_equal(result['A']['C'], expected)
        with pytest.raises(ValueError, match='could not convert string to float'):
            float_string_frame.cov()
        result = float_string_frame.cov(numeric_only=True)
        expected = float_string_frame.loc[:, ['A', 'B', 'C', 'D']].cov()
        tm.assert_frame_equal(result, expected)
        df = DataFrame(np.linspace(0.0, 1.0, 10))
        result = df.cov()
        expected = DataFrame(np.cov(df.values.T).reshape((1, 1)), index=df.columns, columns=df.columns)
        tm.assert_frame_equal(result, expected)
        df.loc[0] = np.nan
        result = df.cov()
        expected = DataFrame(np.cov(df.values[1:].T).reshape((1, 1)), index=df.columns, columns=df.columns)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('test_ddof', [None, 0, 1, 2, 3])
    def test_cov_ddof(self, test_ddof):
        np_array1 = np.random.default_rng(2).random(10)
        np_array2 = np.random.default_rng(2).random(10)
        df = DataFrame({0: np_array1, 1: np_array2})
        result = df.cov(ddof=test_ddof)
        expected_np = np.cov(np_array1, np_array2, ddof=test_ddof)
        expected = DataFrame(expected_np)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('other_column', [pd.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])])
    def test_cov_nullable_integer(self, other_column):
        data = DataFrame({'a': pd.array([1, 2, None]), 'b': other_column})
        result = data.cov()
        arr = np.array([[0.5, 0.5], [0.5, 1.0]])
        expected = DataFrame(arr, columns=['a', 'b'], index=['a', 'b'])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('numeric_only', [True, False])
    def test_cov_numeric_only(self, numeric_only):
        df = DataFrame({'a': [1, 0], 'c': ['x', 'y']})
        expected = DataFrame(0.5, index=['a'], columns=['a'])
        if numeric_only:
            result = df.cov(numeric_only=numeric_only)
            tm.assert_frame_equal(result, expected)
        else:
            with pytest.raises(ValueError, match='could not convert string to float'):
                df.cov(numeric_only=numeric_only)