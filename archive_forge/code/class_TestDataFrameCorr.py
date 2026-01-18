import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestDataFrameCorr:

    @pytest.mark.parametrize('method', ['pearson', 'kendall', 'spearman'])
    def test_corr_scipy_method(self, float_frame, method):
        pytest.importorskip('scipy')
        float_frame.loc[float_frame.index[:5], 'A'] = np.nan
        float_frame.loc[float_frame.index[5:10], 'B'] = np.nan
        float_frame.loc[float_frame.index[:10], 'A'] = float_frame['A'][10:20].copy()
        correls = float_frame.corr(method=method)
        expected = float_frame['A'].corr(float_frame['C'], method=method)
        tm.assert_almost_equal(correls['A']['C'], expected)

    def test_corr_non_numeric(self, float_string_frame):
        with pytest.raises(ValueError, match='could not convert string to float'):
            float_string_frame.corr()
        result = float_string_frame.corr(numeric_only=True)
        expected = float_string_frame.loc[:, ['A', 'B', 'C', 'D']].corr()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('meth', ['pearson', 'kendall', 'spearman'])
    def test_corr_nooverlap(self, meth):
        pytest.importorskip('scipy')
        df = DataFrame({'A': [1, 1.5, 1, np.nan, np.nan, np.nan], 'B': [np.nan, np.nan, np.nan, 1, 1.5, 1], 'C': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]})
        rs = df.corr(meth)
        assert isna(rs.loc['A', 'B'])
        assert isna(rs.loc['B', 'A'])
        assert rs.loc['A', 'A'] == 1
        assert rs.loc['B', 'B'] == 1
        assert isna(rs.loc['C', 'C'])

    @pytest.mark.parametrize('meth', ['pearson', 'spearman'])
    def test_corr_constant(self, meth):
        df = DataFrame({'A': [1, 1, 1, np.nan, np.nan, np.nan], 'B': [np.nan, np.nan, np.nan, 1, 1, 1]})
        rs = df.corr(meth)
        assert isna(rs.values).all()

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    @pytest.mark.parametrize('meth', ['pearson', 'kendall', 'spearman'])
    def test_corr_int_and_boolean(self, meth):
        pytest.importorskip('scipy')
        df = DataFrame({'a': [True, False], 'b': [1, 0]})
        expected = DataFrame(np.ones((2, 2)), index=['a', 'b'], columns=['a', 'b'])
        result = df.corr(meth)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('method', ['cov', 'corr'])
    def test_corr_cov_independent_index_column(self, method):
        df = DataFrame(np.random.default_rng(2).standard_normal(4 * 10).reshape(10, 4), columns=list('abcd'))
        result = getattr(df, method)()
        assert result.index is not result.columns
        assert result.index.equals(result.columns)

    def test_corr_invalid_method(self):
        df = DataFrame(np.random.default_rng(2).normal(size=(10, 2)))
        msg = "method must be either 'pearson', 'spearman', 'kendall', or a callable, "
        with pytest.raises(ValueError, match=msg):
            df.corr(method='____')

    def test_corr_int(self):
        df = DataFrame({'a': [1, 2, 3, 4], 'b': [1, 2, 3, 4]})
        df.cov()
        df.corr()

    @pytest.mark.parametrize('nullable_column', [pd.array([1, 2, 3]), pd.array([1, 2, None])])
    @pytest.mark.parametrize('other_column', [pd.array([1, 2, 3]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, np.nan])])
    @pytest.mark.parametrize('method', ['pearson', 'spearman', 'kendall'])
    def test_corr_nullable_integer(self, nullable_column, other_column, method):
        pytest.importorskip('scipy')
        data = DataFrame({'a': nullable_column, 'b': other_column})
        result = data.corr(method=method)
        expected = DataFrame(np.ones((2, 2)), columns=['a', 'b'], index=['a', 'b'])
        tm.assert_frame_equal(result, expected)

    def test_corr_item_cache(self, using_copy_on_write, warn_copy_on_write):
        df = DataFrame({'A': range(10)})
        df['B'] = range(10)[::-1]
        ser = df['A']
        assert len(df._mgr.arrays) == 2
        _ = df.corr(numeric_only=True)
        if using_copy_on_write:
            ser.iloc[0] = 99
            assert df.loc[0, 'A'] == 0
        else:
            ser.values[0] = 99
            assert df.loc[0, 'A'] == 99
            if not warn_copy_on_write:
                assert df['A'] is ser
            assert df.values[0, 0] == 99

    @pytest.mark.parametrize('length', [2, 20, 200, 2000])
    def test_corr_for_constant_columns(self, length):
        df = DataFrame(length * [[0.4, 0.1]], columns=['A', 'B'])
        result = df.corr()
        expected = DataFrame({'A': [np.nan, np.nan], 'B': [np.nan, np.nan]}, index=['A', 'B'])
        tm.assert_frame_equal(result, expected)

    def test_calc_corr_small_numbers(self):
        df = DataFrame({'A': [1e-20, 2e-20, 3e-20], 'B': [1e-20, 2e-20, 3e-20]})
        result = df.corr()
        expected = DataFrame({'A': [1.0, 1.0], 'B': [1.0, 1.0]}, index=['A', 'B'])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('method', ['pearson', 'spearman', 'kendall'])
    def test_corr_min_periods_greater_than_length(self, method):
        pytest.importorskip('scipy')
        df = DataFrame({'A': [1, 2], 'B': [1, 2]})
        result = df.corr(method=method, min_periods=3)
        expected = DataFrame({'A': [np.nan, np.nan], 'B': [np.nan, np.nan]}, index=['A', 'B'])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('meth', ['pearson', 'kendall', 'spearman'])
    @pytest.mark.parametrize('numeric_only', [True, False])
    def test_corr_numeric_only(self, meth, numeric_only):
        pytest.importorskip('scipy')
        df = DataFrame({'a': [1, 0], 'b': [1, 0], 'c': ['x', 'y']})
        expected = DataFrame(np.ones((2, 2)), index=['a', 'b'], columns=['a', 'b'])
        if numeric_only:
            result = df.corr(meth, numeric_only=numeric_only)
            tm.assert_frame_equal(result, expected)
        else:
            with pytest.raises(ValueError, match='could not convert string to float'):
                df.corr(meth, numeric_only=numeric_only)