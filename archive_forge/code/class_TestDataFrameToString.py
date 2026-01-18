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
class TestDataFrameToString:

    def test_to_string_decimal(self):
        df = DataFrame({'A': [6.0, 3.1, 2.2]})
        expected = '     A\n0  6,0\n1  3,1\n2  2,2'
        assert df.to_string(decimal=',') == expected

    def test_to_string_left_justify_cols(self):
        df = DataFrame({'x': [3234, 0.253]})
        df_s = df.to_string(justify='left')
        expected = '   x       \n0  3234.000\n1     0.253'
        assert df_s == expected

    def test_to_string_format_na(self):
        df = DataFrame({'A': [np.nan, -1, -2.1234, 3, 4], 'B': [np.nan, 'foo', 'foooo', 'fooooo', 'bar']})
        result = df.to_string()
        expected = '        A       B\n0     NaN     NaN\n1 -1.0000     foo\n2 -2.1234   foooo\n3  3.0000  fooooo\n4  4.0000     bar'
        assert result == expected
        df = DataFrame({'A': [np.nan, -1.0, -2.0, 3.0, 4.0], 'B': [np.nan, 'foo', 'foooo', 'fooooo', 'bar']})
        result = df.to_string()
        expected = '     A       B\n0  NaN     NaN\n1 -1.0     foo\n2 -2.0   foooo\n3  3.0  fooooo\n4  4.0     bar'
        assert result == expected

    def test_to_string_with_dict_entries(self):
        df = DataFrame({'A': [{'a': 1, 'b': 2}]})
        val = df.to_string()
        assert "'a': 1" in val
        assert "'b': 2" in val

    def test_to_string_with_categorical_columns(self):
        data = [[4, 2], [3, 2], [4, 3]]
        cols = ['aaaaaaaaa', 'b']
        df = DataFrame(data, columns=cols)
        df_cat_cols = DataFrame(data, columns=CategoricalIndex(cols))
        assert df.to_string() == df_cat_cols.to_string()

    def test_repr_embedded_ndarray(self):
        arr = np.empty(10, dtype=[('err', object)])
        for i in range(len(arr)):
            arr['err'][i] = np.random.default_rng(2).standard_normal(i)
        df = DataFrame(arr)
        repr(df['err'])
        repr(df)
        df.to_string()

    def test_to_string_truncate(self):
        df = DataFrame([{'a': 'foo', 'b': 'bar', 'c': "let's make this a very VERY long line that is longer than the default 50 character limit", 'd': 1}, {'a': 'foo', 'b': 'bar', 'c': 'stuff', 'd': 1}])
        df.set_index(['a', 'b', 'c'])
        assert df.to_string() == "     a    b                                                                                         c  d\n0  foo  bar  let's make this a very VERY long line that is longer than the default 50 character limit  1\n1  foo  bar                                                                                     stuff  1"
        with option_context('max_colwidth', 20):
            assert df.to_string() == "     a    b                                                                                         c  d\n0  foo  bar  let's make this a very VERY long line that is longer than the default 50 character limit  1\n1  foo  bar                                                                                     stuff  1"
        assert df.to_string(max_colwidth=20) == "     a    b                    c  d\n0  foo  bar  let's make this ...  1\n1  foo  bar                stuff  1"

    @pytest.mark.parametrize('input_array, expected', [({'A': ['a']}, 'A\na'), ({'A': ['a', 'b'], 'B': ['c', 'dd']}, 'A  B\na  c\nb dd'), ({'A': ['a', 1], 'B': ['aa', 1]}, 'A  B\na aa\n1  1')])
    def test_format_remove_leading_space_dataframe(self, input_array, expected):
        df = DataFrame(input_array).to_string(index=False)
        assert df == expected

    @pytest.mark.parametrize('data,expected', [({'col1': [1, 2], 'col2': [3, 4]}, '   col1  col2\n0     1     3\n1     2     4'), ({'col1': ['Abc', 0.756], 'col2': [np.nan, 4.5435]}, '    col1    col2\n0    Abc     NaN\n1  0.756  4.5435'), ({'col1': [np.nan, 'a'], 'col2': [0.009, 3.543], 'col3': ['Abc', 23]}, '  col1   col2 col3\n0  NaN  0.009  Abc\n1    a  3.543   23')])
    def test_to_string_max_rows_zero(self, data, expected):
        result = DataFrame(data=data).to_string(max_rows=0)
        assert result == expected

    @pytest.mark.parametrize('max_cols, max_rows, expected', [(10, None, ' 0   1   2   3   4   ...  6   7   8   9   10\n  0   0   0   0   0  ...   0   0   0   0   0\n  0   0   0   0   0  ...   0   0   0   0   0\n  0   0   0   0   0  ...   0   0   0   0   0\n  0   0   0   0   0  ...   0   0   0   0   0'), (None, 2, ' 0   1   2   3   4   5   6   7   8   9   10\n  0   0   0   0   0   0   0   0   0   0   0\n ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..\n  0   0   0   0   0   0   0   0   0   0   0'), (10, 2, ' 0   1   2   3   4   ...  6   7   8   9   10\n  0   0   0   0   0  ...   0   0   0   0   0\n ..  ..  ..  ..  ..  ...  ..  ..  ..  ..  ..\n  0   0   0   0   0  ...   0   0   0   0   0'), (9, 2, ' 0   1   2   3   ...  7   8   9   10\n  0   0   0   0  ...   0   0   0   0\n ..  ..  ..  ..  ...  ..  ..  ..  ..\n  0   0   0   0  ...   0   0   0   0'), (1, 1, ' 0  ...\n 0  ...\n..  ...')])
    def test_truncation_no_index(self, max_cols, max_rows, expected):
        df = DataFrame([[0] * 11] * 4)
        assert df.to_string(index=False, max_cols=max_cols, max_rows=max_rows) == expected

    def test_to_string_no_index(self):
        df = DataFrame({'x': [11, 22], 'y': [33, -44], 'z': ['AAA', '   ']})
        df_s = df.to_string(index=False)
        expected = ' x   y   z\n11  33 AAA\n22 -44    '
        assert df_s == expected
        df_s = df[['y', 'x', 'z']].to_string(index=False)
        expected = '  y  x   z\n 33 11 AAA\n-44 22    '
        assert df_s == expected

    def test_to_string_unicode_columns(self, float_frame):
        df = DataFrame({'σ': np.arange(10.0)})
        buf = StringIO()
        df.to_string(buf=buf)
        buf.getvalue()
        buf = StringIO()
        df.info(buf=buf)
        buf.getvalue()
        result = float_frame.to_string()
        assert isinstance(result, str)

    @pytest.mark.parametrize('na_rep', ['NaN', 'Ted'])
    def test_to_string_na_rep_and_float_format(self, na_rep):
        df = DataFrame([['A', 1.2225], ['A', None]], columns=['Group', 'Data'])
        result = df.to_string(na_rep=na_rep, float_format='{:.2f}'.format)
        expected = dedent(f'               Group  Data\n             0     A  1.22\n             1     A   {na_rep}')
        assert result == expected

    def test_to_string_string_dtype(self):
        pytest.importorskip('pyarrow')
        df = DataFrame({'x': ['foo', 'bar', 'baz'], 'y': ['a', 'b', 'c'], 'z': [1, 2, 3]})
        df = df.astype({'x': 'string[pyarrow]', 'y': 'string[python]', 'z': 'int64[pyarrow]'})
        result = df.dtypes.to_string()
        expected = dedent('            x    string[pyarrow]\n            y     string[python]\n            z     int64[pyarrow]')
        assert result == expected

    def test_to_string_pos_args_deprecation(self):
        df = DataFrame({'a': [1, 2, 3]})
        msg = "Starting with pandas version 3.0 all arguments of to_string except for the argument 'buf' will be keyword-only."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            buf = StringIO()
            df.to_string(buf, None, None, True, True)

    def test_to_string_utf8_columns(self):
        n = 'א'.encode()
        df = DataFrame([1, 2], columns=[n])
        with option_context('display.max_rows', 1):
            repr(df)

    def test_to_string_unicode_two(self):
        dm = DataFrame({'c/σ': []})
        buf = StringIO()
        dm.to_string(buf)

    def test_to_string_unicode_three(self):
        dm = DataFrame(['Â'])
        buf = StringIO()
        dm.to_string(buf)

    def test_to_string_with_float_index(self):
        index = Index([1.5, 2, 3, 4, 5])
        df = DataFrame(np.arange(5), index=index)
        result = df.to_string()
        expected = '     0\n1.5  0\n2.0  1\n3.0  2\n4.0  3\n5.0  4'
        assert result == expected

    def test_to_string(self):
        biggie = DataFrame({'A': np.random.default_rng(2).standard_normal(200), 'B': Index([f'{i}?!' for i in range(200)])})
        biggie.loc[:20, 'A'] = np.nan
        biggie.loc[:20, 'B'] = np.nan
        s = biggie.to_string()
        buf = StringIO()
        retval = biggie.to_string(buf=buf)
        assert retval is None
        assert buf.getvalue() == s
        assert isinstance(s, str)
        result = biggie.to_string(columns=['B', 'A'], col_space=17, float_format='%.5f'.__mod__)
        lines = result.split('\n')
        header = lines[0].strip().split()
        joined = '\n'.join([re.sub('\\s+', ' ', x).strip() for x in lines[1:]])
        recons = read_csv(StringIO(joined), names=header, header=None, sep=' ')
        tm.assert_series_equal(recons['B'], biggie['B'])
        assert recons['A'].count() == biggie['A'].count()
        assert (np.abs(recons['A'].dropna() - biggie['A'].dropna()) < 0.1).all()
        result = biggie.to_string(columns=['A'], col_space=17)
        header = result.split('\n')[0].strip().split()
        expected = ['A']
        assert header == expected
        biggie.to_string(columns=['B', 'A'], formatters={'A': lambda x: f'{x:.1f}'})
        biggie.to_string(columns=['B', 'A'], float_format=str)
        biggie.to_string(columns=['B', 'A'], col_space=12, float_format=str)
        frame = DataFrame(index=np.arange(200))
        frame.to_string()

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason='fix when arrow is default')
    def test_to_string_index_with_nan(self):
        df = DataFrame({'id1': {0: '1a3', 1: '9h4'}, 'id2': {0: np.nan, 1: 'd67'}, 'id3': {0: '78d', 1: '79d'}, 'value': {0: 123, 1: 64}})
        y = df.set_index(['id1', 'id2', 'id3'])
        result = y.to_string()
        expected = '             value\nid1 id2 id3       \n1a3 NaN 78d    123\n9h4 d67 79d     64'
        assert result == expected
        y = df.set_index('id2')
        result = y.to_string()
        expected = '     id1  id3  value\nid2                 \nNaN  1a3  78d    123\nd67  9h4  79d     64'
        assert result == expected
        y = df.set_index(['id1', 'id2']).set_index('id3', append=True)
        result = y.to_string()
        expected = '             value\nid1 id2 id3       \n1a3 NaN 78d    123\n9h4 d67 79d     64'
        assert result == expected
        df2 = df.copy()
        df2.loc[:, 'id2'] = np.nan
        y = df2.set_index('id2')
        result = y.to_string()
        expected = '     id1  id3  value\nid2                 \nNaN  1a3  78d    123\nNaN  9h4  79d     64'
        assert result == expected
        df2 = df.copy()
        df2.loc[:, 'id2'] = np.nan
        y = df2.set_index(['id2', 'id3'])
        result = y.to_string()
        expected = '         id1  value\nid2 id3            \nNaN 78d  1a3    123\n    79d  9h4     64'
        assert result == expected
        df = DataFrame({'id1': {0: np.nan, 1: '9h4'}, 'id2': {0: np.nan, 1: 'd67'}, 'id3': {0: np.nan, 1: '79d'}, 'value': {0: 123, 1: 64}})
        y = df.set_index(['id1', 'id2', 'id3'])
        result = y.to_string()
        expected = '             value\nid1 id2 id3       \nNaN NaN NaN    123\n9h4 d67 79d     64'
        assert result == expected

    def test_to_string_nonunicode_nonascii_alignment(self):
        df = DataFrame([['aaÃ¤Ã¤', 1], ['bbbb', 2]])
        rep_str = df.to_string()
        lines = rep_str.split('\n')
        assert len(lines[1]) == len(lines[2])

    def test_unicode_problem_decoding_as_ascii(self):
        df = DataFrame({'c/σ': Series({'test': np.nan})})
        str(df.to_string())

    def test_to_string_repr_unicode(self):
        buf = StringIO()
        unicode_values = ['σ'] * 10
        unicode_values = np.array(unicode_values, dtype=object)
        df = DataFrame({'unicode': unicode_values})
        df.to_string(col_space=10, buf=buf)
        repr(df)
        _stdin = sys.stdin
        try:
            sys.stdin = None
            repr(df)
        finally:
            sys.stdin = _stdin