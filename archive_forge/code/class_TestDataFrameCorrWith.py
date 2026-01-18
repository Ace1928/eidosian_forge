import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestDataFrameCorrWith:

    @pytest.mark.parametrize('dtype', ['float64', 'Float64', pytest.param('float64[pyarrow]', marks=td.skip_if_no('pyarrow'))])
    def test_corrwith(self, datetime_frame, dtype):
        datetime_frame = datetime_frame.astype(dtype)
        a = datetime_frame
        noise = Series(np.random.default_rng(2).standard_normal(len(a)), index=a.index)
        b = datetime_frame.add(noise, axis=0)
        b = b.reindex(columns=b.columns[::-1], index=b.index[::-1][10:])
        del b['B']
        colcorr = a.corrwith(b, axis=0)
        tm.assert_almost_equal(colcorr['A'], a['A'].corr(b['A']))
        rowcorr = a.corrwith(b, axis=1)
        tm.assert_series_equal(rowcorr, a.T.corrwith(b.T, axis=0))
        dropped = a.corrwith(b, axis=0, drop=True)
        tm.assert_almost_equal(dropped['A'], a['A'].corr(b['A']))
        assert 'B' not in dropped
        dropped = a.corrwith(b, axis=1, drop=True)
        assert a.index[-1] not in dropped.index
        index = ['a', 'b', 'c', 'd', 'e']
        columns = ['one', 'two', 'three', 'four']
        df1 = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), index=index, columns=columns)
        df2 = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), index=index[:4], columns=columns)
        correls = df1.corrwith(df2, axis=1)
        for row in index[:4]:
            tm.assert_almost_equal(correls[row], df1.loc[row].corr(df2.loc[row]))

    def test_corrwith_with_objects(self, using_infer_string):
        df1 = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        df2 = df1.copy()
        cols = ['A', 'B', 'C', 'D']
        df1['obj'] = 'foo'
        df2['obj'] = 'bar'
        if using_infer_string:
            import pyarrow as pa
            with pytest.raises(pa.lib.ArrowNotImplementedError, match='has no kernel'):
                df1.corrwith(df2)
        else:
            with pytest.raises(TypeError, match='Could not convert'):
                df1.corrwith(df2)
        result = df1.corrwith(df2, numeric_only=True)
        expected = df1.loc[:, cols].corrwith(df2.loc[:, cols])
        tm.assert_series_equal(result, expected)
        with pytest.raises(TypeError, match='unsupported operand type'):
            df1.corrwith(df2, axis=1)
        result = df1.corrwith(df2, axis=1, numeric_only=True)
        expected = df1.loc[:, cols].corrwith(df2.loc[:, cols], axis=1)
        tm.assert_series_equal(result, expected)

    def test_corrwith_series(self, datetime_frame):
        result = datetime_frame.corrwith(datetime_frame['A'])
        expected = datetime_frame.apply(datetime_frame['A'].corr)
        tm.assert_series_equal(result, expected)

    def test_corrwith_matches_corrcoef(self):
        df1 = DataFrame(np.arange(10000), columns=['a'])
        df2 = DataFrame(np.arange(10000) ** 2, columns=['a'])
        c1 = df1.corrwith(df2)['a']
        c2 = np.corrcoef(df1['a'], df2['a'])[0][1]
        tm.assert_almost_equal(c1, c2)
        assert c1 < 1

    @pytest.mark.parametrize('numeric_only', [True, False])
    def test_corrwith_mixed_dtypes(self, numeric_only):
        df = DataFrame({'a': [1, 4, 3, 2], 'b': [4, 6, 7, 3], 'c': ['a', 'b', 'c', 'd']})
        s = Series([0, 6, 7, 3])
        if numeric_only:
            result = df.corrwith(s, numeric_only=numeric_only)
            corrs = [df['a'].corr(s), df['b'].corr(s)]
            expected = Series(data=corrs, index=['a', 'b'])
            tm.assert_series_equal(result, expected)
        else:
            with pytest.raises(ValueError, match='could not convert string to float'):
                df.corrwith(s, numeric_only=numeric_only)

    def test_corrwith_index_intersection(self):
        df1 = DataFrame(np.random.default_rng(2).random(size=(10, 2)), columns=['a', 'b'])
        df2 = DataFrame(np.random.default_rng(2).random(size=(10, 3)), columns=['a', 'b', 'c'])
        result = df1.corrwith(df2, drop=True).index.sort_values()
        expected = df1.columns.intersection(df2.columns).sort_values()
        tm.assert_index_equal(result, expected)

    def test_corrwith_index_union(self):
        df1 = DataFrame(np.random.default_rng(2).random(size=(10, 2)), columns=['a', 'b'])
        df2 = DataFrame(np.random.default_rng(2).random(size=(10, 3)), columns=['a', 'b', 'c'])
        result = df1.corrwith(df2, drop=False).index.sort_values()
        expected = df1.columns.union(df2.columns).sort_values()
        tm.assert_index_equal(result, expected)

    def test_corrwith_dup_cols(self):
        df1 = DataFrame(np.vstack([np.arange(10)] * 3).T)
        df2 = df1.copy()
        df2 = pd.concat((df2, df2[0]), axis=1)
        result = df1.corrwith(df2)
        expected = Series(np.ones(4), index=[0, 0, 1, 2])
        tm.assert_series_equal(result, expected)

    def test_corr_numerical_instabilities(self):
        df = DataFrame([[0.2, 0.4], [0.4, 0.2]])
        result = df.corr()
        expected = DataFrame({0: [1.0, -1.0], 1: [-1.0, 1.0]})
        tm.assert_frame_equal(result - 1, expected - 1, atol=1e-17)

    def test_corrwith_spearman(self):
        pytest.importorskip('scipy')
        df = DataFrame(np.random.default_rng(2).random(size=(100, 3)))
        result = df.corrwith(df ** 2, method='spearman')
        expected = Series(np.ones(len(result)))
        tm.assert_series_equal(result, expected)

    def test_corrwith_kendall(self):
        pytest.importorskip('scipy')
        df = DataFrame(np.random.default_rng(2).random(size=(100, 3)))
        result = df.corrwith(df ** 2, method='kendall')
        expected = Series(np.ones(len(result)))
        tm.assert_series_equal(result, expected)

    def test_corrwith_spearman_with_tied_data(self):
        pytest.importorskip('scipy')
        df1 = DataFrame({'A': [1, np.nan, 7, 8], 'B': [False, True, True, False], 'C': [10, 4, 9, 3]})
        df2 = df1[['B', 'C']]
        result = (df1 + 1).corrwith(df2.B, method='spearman')
        expected = Series([0.0, 1.0, 0.0], index=['A', 'B', 'C'])
        tm.assert_series_equal(result, expected)
        df_bool = DataFrame({'A': [True, True, False, False], 'B': [True, False, False, True]})
        ser_bool = Series([True, True, False, True])
        result = df_bool.corrwith(ser_bool)
        expected = Series([0.57735, 0.57735], index=['A', 'B'])
        tm.assert_series_equal(result, expected)