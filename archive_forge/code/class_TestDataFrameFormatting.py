from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
class TestDataFrameFormatting:

    def test_repr_truncation(self):
        max_len = 20
        with option_context('display.max_colwidth', max_len):
            df = DataFrame({'A': np.random.default_rng(2).standard_normal(10), 'B': ['a' * np.random.default_rng(2).integers(max_len - 1, max_len + 1) for _ in range(10)]})
            r = repr(df)
            r = r[r.find('\n') + 1:]
            adj = printing.get_adjustment()
            for line, value in zip(r.split('\n'), df['B']):
                if adj.len(value) + 1 > max_len:
                    assert '...' in line
                else:
                    assert '...' not in line
        with option_context('display.max_colwidth', 999999):
            assert '...' not in repr(df)
        with option_context('display.max_colwidth', max_len + 2):
            assert '...' not in repr(df)

    def test_repr_truncation_preserves_na(self):
        df = DataFrame({'a': [pd.NA for _ in range(10)]})
        with option_context('display.max_rows', 2, 'display.show_dimensions', False):
            assert repr(df) == '       a\n0   <NA>\n..   ...\n9   <NA>'

    def test_max_colwidth_negative_int_raises(self):
        with pytest.raises(ValueError, match='Value must be a nonnegative integer or None'):
            with option_context('display.max_colwidth', -1):
                pass

    def test_repr_chop_threshold(self):
        df = DataFrame([[0.1, 0.5], [0.5, -0.1]])
        reset_option('display.chop_threshold')
        assert repr(df) == '     0    1\n0  0.1  0.5\n1  0.5 -0.1'
        with option_context('display.chop_threshold', 0.2):
            assert repr(df) == '     0    1\n0  0.0  0.5\n1  0.5  0.0'
        with option_context('display.chop_threshold', 0.6):
            assert repr(df) == '     0    1\n0  0.0  0.0\n1  0.0  0.0'
        with option_context('display.chop_threshold', None):
            assert repr(df) == '     0    1\n0  0.1  0.5\n1  0.5 -0.1'

    def test_repr_chop_threshold_column_below(self):
        df = DataFrame([[10, 20, 30, 40], [8e-10, -1e-11, 2e-09, -2e-11]]).T
        with option_context('display.chop_threshold', 0):
            assert repr(df) == '      0             1\n0  10.0  8.000000e-10\n1  20.0 -1.000000e-11\n2  30.0  2.000000e-09\n3  40.0 -2.000000e-11'
        with option_context('display.chop_threshold', 1e-08):
            assert repr(df) == '      0             1\n0  10.0  0.000000e+00\n1  20.0  0.000000e+00\n2  30.0  0.000000e+00\n3  40.0  0.000000e+00'
        with option_context('display.chop_threshold', 5e-11):
            assert repr(df) == '      0             1\n0  10.0  8.000000e-10\n1  20.0  0.000000e+00\n2  30.0  2.000000e-09\n3  40.0  0.000000e+00'

    def test_repr_no_backslash(self):
        with option_context('mode.sim_interactive', True):
            df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
            assert '\\' not in repr(df)

    def test_expand_frame_repr(self):
        df_small = DataFrame('hello', index=[0], columns=[0])
        df_wide = DataFrame('hello', index=[0], columns=range(10))
        df_tall = DataFrame('hello', index=range(30), columns=range(5))
        with option_context('mode.sim_interactive', True):
            with option_context('display.max_columns', 10, 'display.width', 20, 'display.max_rows', 20, 'display.show_dimensions', True):
                with option_context('display.expand_frame_repr', True):
                    assert not has_truncated_repr(df_small)
                    assert not has_expanded_repr(df_small)
                    assert not has_truncated_repr(df_wide)
                    assert has_expanded_repr(df_wide)
                    assert has_vertically_truncated_repr(df_tall)
                    assert has_expanded_repr(df_tall)
                with option_context('display.expand_frame_repr', False):
                    assert not has_truncated_repr(df_small)
                    assert not has_expanded_repr(df_small)
                    assert not has_horizontally_truncated_repr(df_wide)
                    assert not has_expanded_repr(df_wide)
                    assert has_vertically_truncated_repr(df_tall)
                    assert not has_expanded_repr(df_tall)

    def test_repr_non_interactive(self):
        df = DataFrame('hello', index=range(1000), columns=range(5))
        with option_context('mode.sim_interactive', False, 'display.width', 0, 'display.max_rows', 5000):
            assert not has_truncated_repr(df)
            assert not has_expanded_repr(df)

    def test_repr_truncates_terminal_size(self, monkeypatch):
        terminal_size = (118, 96)
        monkeypatch.setattr('pandas.io.formats.format.get_terminal_size', lambda: terminal_size)
        index = range(5)
        columns = MultiIndex.from_tuples([('This is a long title with > 37 chars.', 'cat'), ('This is a loooooonger title with > 43 chars.', 'dog')])
        df = DataFrame(1, index=index, columns=columns)
        result = repr(df)
        h1, h2 = result.split('\n')[:2]
        assert 'long' in h1
        assert 'loooooonger' in h1
        assert 'cat' in h2
        assert 'dog' in h2
        df2 = DataFrame({'A' * 41: [1, 2], 'B' * 41: [1, 2]})
        result = repr(df2)
        assert df2.columns[0] in result.split('\n')[0]

    def test_repr_truncates_terminal_size_full(self, monkeypatch):
        terminal_size = (80, 24)
        df = DataFrame(np.random.default_rng(2).random((1, 7)))
        monkeypatch.setattr('pandas.io.formats.format.get_terminal_size', lambda: terminal_size)
        assert '...' not in str(df)

    def test_repr_truncation_column_size(self):
        df = DataFrame({'a': [108480, 30830], 'b': [12345, 12345], 'c': [12345, 12345], 'd': [12345, 12345], 'e': ['a' * 50] * 2})
        assert '...' in str(df)
        assert '    ...    ' not in str(df)

    def test_repr_max_columns_max_rows(self):
        term_width, term_height = get_terminal_size()
        if term_width < 10 or term_height < 10:
            pytest.skip(f'terminal size too small, {term_width} x {term_height}')

        def mkframe(n):
            index = [f'{i:05d}' for i in range(n)]
            return DataFrame(0, index, index)
        df6 = mkframe(6)
        df10 = mkframe(10)
        with option_context('mode.sim_interactive', True):
            with option_context('display.width', term_width * 2):
                with option_context('display.max_rows', 5, 'display.max_columns', 5):
                    assert not has_expanded_repr(mkframe(4))
                    assert not has_expanded_repr(mkframe(5))
                    assert not has_expanded_repr(df6)
                    assert has_doubly_truncated_repr(df6)
                with option_context('display.max_rows', 20, 'display.max_columns', 10):
                    assert not has_expanded_repr(df6)
                    assert not has_truncated_repr(df6)
                with option_context('display.max_rows', 9, 'display.max_columns', 10):
                    assert not has_expanded_repr(df10)
                    assert has_vertically_truncated_repr(df10)
            with option_context('display.max_columns', 100, 'display.max_rows', term_width * 20, 'display.width', None):
                df = mkframe(term_width // 7 - 2)
                assert not has_expanded_repr(df)
                df = mkframe(term_width // 7 + 2)
                printing.pprint_thing(df._repr_fits_horizontal_())
                assert has_expanded_repr(df)

    def test_repr_min_rows(self):
        df = DataFrame({'a': range(20)})
        assert '..' not in repr(df)
        assert '..' not in df._repr_html_()
        df = DataFrame({'a': range(61)})
        assert '..' in repr(df)
        assert '..' in df._repr_html_()
        with option_context('display.max_rows', 10, 'display.min_rows', 4):
            assert '..' in repr(df)
            assert '2  ' not in repr(df)
            assert '...' in df._repr_html_()
            assert '<td>2</td>' not in df._repr_html_()
        with option_context('display.max_rows', 12, 'display.min_rows', None):
            assert '5    5' in repr(df)
            assert '<td>5</td>' in df._repr_html_()
        with option_context('display.max_rows', 10, 'display.min_rows', 12):
            assert '5    5' not in repr(df)
            assert '<td>5</td>' not in df._repr_html_()
        with option_context('display.max_rows', None, 'display.min_rows', 12):
            assert '..' not in repr(df)
            assert '..' not in df._repr_html_()

    def test_str_max_colwidth(self):
        df = DataFrame([{'a': 'foo', 'b': 'bar', 'c': 'uncomfortably long line with lots of stuff', 'd': 1}, {'a': 'foo', 'b': 'bar', 'c': 'stuff', 'd': 1}])
        df.set_index(['a', 'b', 'c'])
        assert str(df) == '     a    b                                           c  d\n0  foo  bar  uncomfortably long line with lots of stuff  1\n1  foo  bar                                       stuff  1'
        with option_context('max_colwidth', 20):
            assert str(df) == '     a    b                    c  d\n0  foo  bar  uncomfortably lo...  1\n1  foo  bar                stuff  1'

    def test_auto_detect(self):
        term_width, term_height = get_terminal_size()
        fac = 1.05
        cols = range(int(term_width * fac))
        index = range(10)
        df = DataFrame(index=index, columns=cols)
        with option_context('mode.sim_interactive', True):
            with option_context('display.max_rows', None):
                with option_context('display.max_columns', None):
                    assert has_expanded_repr(df)
            with option_context('display.max_rows', 0):
                with option_context('display.max_columns', 0):
                    assert has_horizontally_truncated_repr(df)
            index = range(int(term_height * fac))
            df = DataFrame(index=index, columns=cols)
            with option_context('display.max_rows', 0):
                with option_context('display.max_columns', None):
                    assert has_expanded_repr(df)
                    assert has_vertically_truncated_repr(df)
            with option_context('display.max_rows', None):
                with option_context('display.max_columns', 0):
                    assert has_horizontally_truncated_repr(df)

    def test_to_string_repr_unicode2(self):
        idx = Index(['abc', 'σa', 'aegdvg'])
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        rs = repr(ser).split('\n')
        line_len = len(rs[0])
        for line in rs[1:]:
            try:
                line = line.decode(get_option('display.encoding'))
            except AttributeError:
                pass
            if not line.startswith('dtype:'):
                assert len(line) == line_len

    def test_east_asian_unicode_false(self):
        df = DataFrame({'a': ['あ', 'いいい', 'う', 'ええええええ'], 'b': [1, 222, 33333, 4]}, index=['a', 'bb', 'c', 'ddd'])
        expected = '          a      b\na         あ      1\nbb      いいい    222\nc         う  33333\nddd  ええええええ      4'
        assert repr(df) == expected
        df = DataFrame({'a': [1, 222, 33333, 4], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=['a', 'bb', 'c', 'ddd'])
        expected = '         a       b\na        1       あ\nbb     222     いいい\nc    33333       う\nddd      4  ええええええ'
        assert repr(df) == expected
        df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=['a', 'bb', 'c', 'ddd'])
        expected = '         a       b\na    あああああ       あ\nbb       い     いいい\nc        う       う\nddd    えええ  ええええええ'
        assert repr(df) == expected
        df = DataFrame({'b': ['あ', 'いいい', 'う', 'ええええええ'], 'あああああ': [1, 222, 33333, 4]}, index=['a', 'bb', 'c', 'ddd'])
        expected = '          b  あああああ\na         あ      1\nbb      いいい    222\nc         う  33333\nddd  ええええええ      4'
        assert repr(df) == expected
        df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=['あああ', 'いいいいいい', 'うう', 'え'])
        expected = '            a       b\nあああ     あああああ       あ\nいいいいいい      い     いいい\nうう          う       う\nえ         えええ  ええええええ'
        assert repr(df) == expected
        df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=Index(['あ', 'い', 'うう', 'え'], name='おおおお'))
        expected = '          a       b\nおおおお               \nあ     あああああ       あ\nい         い     いいい\nうう        う       う\nえ       えええ  ええええええ'
        assert repr(df) == expected
        df = DataFrame({'あああ': ['あああ', 'い', 'う', 'えええええ'], 'いいいいい': ['あ', 'いいい', 'う', 'ええ']}, index=Index(['あ', 'いいい', 'うう', 'え'], name='お'))
        expected = '       あああ いいいいい\nお               \nあ      あああ     あ\nいいい      い   いいい\nうう       う     う\nえ    えええええ    ええ'
        assert repr(df) == expected
        idx = MultiIndex.from_tuples([('あ', 'いい'), ('う', 'え'), ('おおお', 'かかかか'), ('き', 'くく')])
        df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=idx)
        expected = '              a       b\nあ   いい    あああああ       あ\nう   え         い     いいい\nおおお かかかか      う       う\nき   くく      えええ  ええええええ'
        assert repr(df) == expected
        with option_context('display.max_rows', 3, 'display.max_columns', 3):
            df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ'], 'c': ['お', 'か', 'ききき', 'くくくくくく'], 'ああああ': ['さ', 'し', 'す', 'せ']}, columns=['a', 'b', 'c', 'ああああ'])
            expected = '        a  ... ああああ\n0   あああああ  ...    さ\n..    ...  ...  ...\n3     えええ  ...    せ\n\n[4 rows x 4 columns]'
            assert repr(df) == expected
            df.index = ['あああ', 'いいいい', 'う', 'aaa']
            expected = '         a  ... ああああ\nあああ  あああああ  ...    さ\n..     ...  ...  ...\naaa    えええ  ...    せ\n\n[4 rows x 4 columns]'
            assert repr(df) == expected

    def test_east_asian_unicode_true(self):
        with option_context('display.unicode.east_asian_width', True):
            df = DataFrame({'a': ['あ', 'いいい', 'う', 'ええええええ'], 'b': [1, 222, 33333, 4]}, index=['a', 'bb', 'c', 'ddd'])
            expected = '                a      b\na              あ      1\nbb         いいい    222\nc              う  33333\nddd  ええええええ      4'
            assert repr(df) == expected
            df = DataFrame({'a': [1, 222, 33333, 4], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=['a', 'bb', 'c', 'ddd'])
            expected = '         a             b\na        1            あ\nbb     222        いいい\nc    33333            う\nddd      4  ええええええ'
            assert repr(df) == expected
            df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=['a', 'bb', 'c', 'ddd'])
            expected = '              a             b\na    あああああ            あ\nbb           い        いいい\nc            う            う\nddd      えええ  ええええええ'
            assert repr(df) == expected
            df = DataFrame({'b': ['あ', 'いいい', 'う', 'ええええええ'], 'あああああ': [1, 222, 33333, 4]}, index=['a', 'bb', 'c', 'ddd'])
            expected = '                b  あああああ\na              あ           1\nbb         いいい         222\nc              う       33333\nddd  ええええええ           4'
            assert repr(df) == expected
            df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=['あああ', 'いいいいいい', 'うう', 'え'])
            expected = '                       a             b\nあああ        あああああ            あ\nいいいいいい          い        いいい\nうう                  う            う\nえ                えええ  ええええええ'
            assert repr(df) == expected
            df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=Index(['あ', 'い', 'うう', 'え'], name='おおおお'))
            expected = '                   a             b\nおおおお                          \nあ        あああああ            あ\nい                い        いいい\nうう              う            う\nえ            えええ  ええええええ'
            assert repr(df) == expected
            df = DataFrame({'あああ': ['あああ', 'い', 'う', 'えええええ'], 'いいいいい': ['あ', 'いいい', 'う', 'ええ']}, index=Index(['あ', 'いいい', 'うう', 'え'], name='お'))
            expected = '            あああ いいいいい\nお                           \nあ          あああ         あ\nいいい          い     いいい\nうう            う         う\nえ      えええええ       ええ'
            assert repr(df) == expected
            idx = MultiIndex.from_tuples([('あ', 'いい'), ('う', 'え'), ('おおお', 'かかかか'), ('き', 'くく')])
            df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=idx)
            expected = '                          a             b\nあ     いい      あああああ            あ\nう     え                い        いいい\nおおお かかかか          う            う\nき     くく          えええ  ええええええ'
            assert repr(df) == expected
            with option_context('display.max_rows', 3, 'display.max_columns', 3):
                df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ'], 'c': ['お', 'か', 'ききき', 'くくくくくく'], 'ああああ': ['さ', 'し', 'す', 'せ']}, columns=['a', 'b', 'c', 'ああああ'])
                expected = '             a  ... ああああ\n0   あああああ  ...       さ\n..         ...  ...      ...\n3       えええ  ...       せ\n\n[4 rows x 4 columns]'
                assert repr(df) == expected
                df.index = ['あああ', 'いいいい', 'う', 'aaa']
                expected = '                 a  ... ああああ\nあああ  あああああ  ...       さ\n...            ...  ...      ...\naaa         えええ  ...       せ\n\n[4 rows x 4 columns]'
                assert repr(df) == expected
            df = DataFrame({'b': ['あ', 'いいい', '¡¡', 'ええええええ'], 'あああああ': [1, 222, 33333, 4]}, index=['a', 'bb', 'c', '¡¡¡'])
            expected = '                b  あああああ\na              あ           1\nbb         いいい         222\nc              ¡¡       33333\n¡¡¡  ええええええ           4'
            assert repr(df) == expected

    def test_to_string_buffer_all_unicode(self):
        buf = StringIO()
        empty = DataFrame({'c/σ': Series(dtype=object)})
        nonempty = DataFrame({'c/σ': Series([1, 2, 3])})
        print(empty, file=buf)
        print(nonempty, file=buf)
        buf.getvalue()

    @pytest.mark.parametrize('index_scalar', ['a' * 10, 1, Timestamp(2020, 1, 1), pd.Period('2020-01-01')])
    @pytest.mark.parametrize('h', [10, 20])
    @pytest.mark.parametrize('w', [10, 20])
    def test_to_string_truncate_indices(self, index_scalar, h, w):
        with option_context('display.expand_frame_repr', False):
            df = DataFrame(index=[index_scalar] * h, columns=[str(i) * 10 for i in range(w)])
            with option_context('display.max_rows', 15):
                if h == 20:
                    assert has_vertically_truncated_repr(df)
                else:
                    assert not has_vertically_truncated_repr(df)
            with option_context('display.max_columns', 15):
                if w == 20:
                    assert has_horizontally_truncated_repr(df)
                else:
                    assert not has_horizontally_truncated_repr(df)
            with option_context('display.max_rows', 15, 'display.max_columns', 15):
                if h == 20 and w == 20:
                    assert has_doubly_truncated_repr(df)
                else:
                    assert not has_doubly_truncated_repr(df)

    def test_to_string_truncate_multilevel(self):
        arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'], ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
        df = DataFrame(index=arrays, columns=arrays)
        with option_context('display.max_rows', 7, 'display.max_columns', 7):
            assert has_doubly_truncated_repr(df)

    @pytest.mark.parametrize('dtype', ['object', 'datetime64[us]'])
    def test_truncate_with_different_dtypes(self, dtype):
        ser = Series([datetime(2012, 1, 1)] * 10 + [datetime(1012, 1, 2)] + [datetime(2012, 1, 3)] * 10, dtype=dtype)
        with option_context('display.max_rows', 8):
            result = str(ser)
        assert dtype in result

    def test_truncate_with_different_dtypes2(self):
        df = DataFrame({'text': ['some words'] + [None] * 9}, dtype=object)
        with option_context('display.max_rows', 8, 'display.max_columns', 3):
            result = str(df)
            assert 'None' in result
            assert 'NaN' not in result

    def test_truncate_with_different_dtypes_multiindex(self):
        df = DataFrame({'Vals': range(100)})
        frame = pd.concat([df], keys=['Sweep'], names=['Sweep', 'Index'])
        result = repr(frame)
        result2 = repr(frame.iloc[:5])
        assert result.startswith(result2)

    def test_datetimelike_frame(self):
        df = DataFrame({'date': [Timestamp('20130101').tz_localize('UTC')] + [NaT] * 5})
        with option_context('display.max_rows', 5):
            result = str(df)
            assert '2013-01-01 00:00:00+00:00' in result
            assert 'NaT' in result
            assert '...' in result
            assert '[6 rows x 1 columns]' in result
        dts = [Timestamp('2011-01-01', tz='US/Eastern')] * 5 + [NaT] * 5
        df = DataFrame({'dt': dts, 'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        with option_context('display.max_rows', 5):
            expected = '                          dt   x\n0  2011-01-01 00:00:00-05:00   1\n1  2011-01-01 00:00:00-05:00   2\n..                       ...  ..\n8                        NaT   9\n9                        NaT  10\n\n[10 rows x 2 columns]'
            assert repr(df) == expected
        dts = [NaT] * 5 + [Timestamp('2011-01-01', tz='US/Eastern')] * 5
        df = DataFrame({'dt': dts, 'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        with option_context('display.max_rows', 5):
            expected = '                          dt   x\n0                        NaT   1\n1                        NaT   2\n..                       ...  ..\n8  2011-01-01 00:00:00-05:00   9\n9  2011-01-01 00:00:00-05:00  10\n\n[10 rows x 2 columns]'
            assert repr(df) == expected
        dts = [Timestamp('2011-01-01', tz='Asia/Tokyo')] * 5 + [Timestamp('2011-01-01', tz='US/Eastern')] * 5
        df = DataFrame({'dt': dts, 'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        with option_context('display.max_rows', 5):
            expected = '                           dt   x\n0   2011-01-01 00:00:00+09:00   1\n1   2011-01-01 00:00:00+09:00   2\n..                        ...  ..\n8   2011-01-01 00:00:00-05:00   9\n9   2011-01-01 00:00:00-05:00  10\n\n[10 rows x 2 columns]'
            assert repr(df) == expected

    @pytest.mark.parametrize('start_date', ['2017-01-01 23:59:59.999999999', '2017-01-01 23:59:59.99999999', '2017-01-01 23:59:59.9999999', '2017-01-01 23:59:59.999999', '2017-01-01 23:59:59.99999', '2017-01-01 23:59:59.9999'])
    def test_datetimeindex_highprecision(self, start_date):
        df = DataFrame({'A': date_range(start=start_date, freq='D', periods=5)})
        result = str(df)
        assert start_date in result
        dti = date_range(start=start_date, freq='D', periods=5)
        df = DataFrame({'A': range(5)}, index=dti)
        result = str(df.index)
        assert start_date in result

    def test_string_repr_encoding(self, datapath):
        filepath = datapath('io', 'parser', 'data', 'unicode_series.csv')
        df = read_csv(filepath, header=None, encoding='latin1')
        repr(df)
        repr(df[1])

    def test_repr_corner(self):
        df = DataFrame({'foo': [-np.inf, np.inf]})
        repr(df)

    def test_frame_info_encoding(self):
        index = ["'Til There Was You (1997)", 'ldum klaka (Cold Fever) (1994)']
        with option_context('display.max_rows', 1):
            df = DataFrame(columns=['a', 'b', 'c'], index=index)
            repr(df)
            repr(df.T)

    def test_wide_repr(self):
        with option_context('mode.sim_interactive', True, 'display.show_dimensions', True, 'display.max_columns', 20):
            max_cols = get_option('display.max_columns')
            df = DataFrame([['a' * 25] * (max_cols - 1)] * 10)
            with option_context('display.expand_frame_repr', False):
                rep_str = repr(df)
            assert f'10 rows x {max_cols - 1} columns' in rep_str
            with option_context('display.expand_frame_repr', True):
                wide_repr = repr(df)
            assert rep_str != wide_repr
            with option_context('display.width', 120):
                wider_repr = repr(df)
                assert len(wider_repr) < len(wide_repr)

    def test_wide_repr_wide_columns(self):
        with option_context('mode.sim_interactive', True, 'display.max_columns', 20):
            df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), columns=['a' * 90, 'b' * 90, 'c' * 90])
            rep_str = repr(df)
            assert len(rep_str.splitlines()) == 20

    def test_wide_repr_named(self):
        with option_context('mode.sim_interactive', True, 'display.max_columns', 20):
            max_cols = get_option('display.max_columns')
            df = DataFrame([['a' * 25] * (max_cols - 1)] * 10)
            df.index.name = 'DataFrame Index'
            with option_context('display.expand_frame_repr', False):
                rep_str = repr(df)
            with option_context('display.expand_frame_repr', True):
                wide_repr = repr(df)
            assert rep_str != wide_repr
            with option_context('display.width', 150):
                wider_repr = repr(df)
                assert len(wider_repr) < len(wide_repr)
            for line in wide_repr.splitlines()[1::13]:
                assert 'DataFrame Index' in line

    def test_wide_repr_multiindex(self):
        with option_context('mode.sim_interactive', True, 'display.max_columns', 20):
            midx = MultiIndex.from_arrays([['a' * 5] * 10] * 2)
            max_cols = get_option('display.max_columns')
            df = DataFrame([['a' * 25] * (max_cols - 1)] * 10, index=midx)
            df.index.names = ['Level 0', 'Level 1']
            with option_context('display.expand_frame_repr', False):
                rep_str = repr(df)
            with option_context('display.expand_frame_repr', True):
                wide_repr = repr(df)
            assert rep_str != wide_repr
            with option_context('display.width', 150):
                wider_repr = repr(df)
                assert len(wider_repr) < len(wide_repr)
            for line in wide_repr.splitlines()[1::13]:
                assert 'Level 0 Level 1' in line

    def test_wide_repr_multiindex_cols(self):
        with option_context('mode.sim_interactive', True, 'display.max_columns', 20):
            max_cols = get_option('display.max_columns')
            midx = MultiIndex.from_arrays([['a' * 5] * 10] * 2)
            mcols = MultiIndex.from_arrays([['b' * 3] * (max_cols - 1)] * 2)
            df = DataFrame([['c' * 25] * (max_cols - 1)] * 10, index=midx, columns=mcols)
            df.index.names = ['Level 0', 'Level 1']
            with option_context('display.expand_frame_repr', False):
                rep_str = repr(df)
            with option_context('display.expand_frame_repr', True):
                wide_repr = repr(df)
            assert rep_str != wide_repr
        with option_context('display.width', 150, 'display.max_columns', 20):
            wider_repr = repr(df)
            assert len(wider_repr) < len(wide_repr)

    def test_wide_repr_unicode(self):
        with option_context('mode.sim_interactive', True, 'display.max_columns', 20):
            max_cols = 20
            df = DataFrame([['a' * 25] * 10] * (max_cols - 1))
            with option_context('display.expand_frame_repr', False):
                rep_str = repr(df)
            with option_context('display.expand_frame_repr', True):
                wide_repr = repr(df)
            assert rep_str != wide_repr
            with option_context('display.width', 150):
                wider_repr = repr(df)
                assert len(wider_repr) < len(wide_repr)

    def test_wide_repr_wide_long_columns(self):
        with option_context('mode.sim_interactive', True):
            df = DataFrame({'a': ['a' * 30, 'b' * 30], 'b': ['c' * 70, 'd' * 80]})
            result = repr(df)
            assert 'ccccc' in result
            assert 'ddddd' in result

    def test_long_series(self):
        n = 1000
        s = Series(np.random.default_rng(2).integers(-50, 50, n), index=[f's{x:04d}' for x in range(n)], dtype='int64')
        str_rep = str(s)
        nmatches = len(re.findall('dtype', str_rep))
        assert nmatches == 1

    def test_to_string_ascii_error(self):
        data = [('0  ', '                        .gitignore ', '     5 ', ' â\x80¢â\x80¢â\x80¢â\x80¢â\x80¢')]
        df = DataFrame(data)
        repr(df)

    def test_show_dimensions(self):
        df = DataFrame(123, index=range(10, 15), columns=range(30))
        with option_context('display.max_rows', 10, 'display.max_columns', 40, 'display.width', 500, 'display.expand_frame_repr', 'info', 'display.show_dimensions', True):
            assert '5 rows' in str(df)
            assert '5 rows' in df._repr_html_()
        with option_context('display.max_rows', 10, 'display.max_columns', 40, 'display.width', 500, 'display.expand_frame_repr', 'info', 'display.show_dimensions', False):
            assert '5 rows' not in str(df)
            assert '5 rows' not in df._repr_html_()
        with option_context('display.max_rows', 2, 'display.max_columns', 2, 'display.width', 500, 'display.expand_frame_repr', 'info', 'display.show_dimensions', 'truncate'):
            assert '5 rows' in str(df)
            assert '5 rows' in df._repr_html_()
        with option_context('display.max_rows', 10, 'display.max_columns', 40, 'display.width', 500, 'display.expand_frame_repr', 'info', 'display.show_dimensions', 'truncate'):
            assert '5 rows' not in str(df)
            assert '5 rows' not in df._repr_html_()

    def test_info_repr(self):
        term_width, term_height = get_terminal_size()
        max_rows = 60
        max_cols = 20 + (max(term_width, 80) - 80) // 4
        h, w = (max_rows + 1, max_cols - 1)
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        assert has_vertically_truncated_repr(df)
        with option_context('display.large_repr', 'info'):
            assert has_info_repr(df)
        h, w = (max_rows - 1, max_cols + 1)
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        assert has_horizontally_truncated_repr(df)
        with option_context('display.large_repr', 'info', 'display.max_columns', max_cols):
            assert has_info_repr(df)

    def test_info_repr_max_cols(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        with option_context('display.large_repr', 'info', 'display.max_columns', 1, 'display.max_info_columns', 4):
            assert has_non_verbose_info_repr(df)
        with option_context('display.large_repr', 'info', 'display.max_columns', 1, 'display.max_info_columns', 5):
            assert not has_non_verbose_info_repr(df)

    def test_pprint_pathological_object(self):
        """
        If the test fails, it at least won't hang.
        """

        class A:

            def __getitem__(self, key):
                return 3
        df = DataFrame([A()])
        repr(df)

    def test_float_trim_zeros(self):
        vals = [20843091730.5, 35220501730.5, 23067481730.5, 20395421730.5, 55989781730.5]
        skip = True
        for line in repr(DataFrame({'A': vals})).split('\n')[:-2]:
            if line.startswith('dtype:'):
                continue
            if _three_digit_exp():
                assert '+010' in line or skip
            else:
                assert '+10' in line or skip
            skip = False

    @pytest.mark.parametrize('data, expected', [(['3.50'], '0    3.50\ndtype: object'), ([1.2, '1.00'], '0     1.2\n1    1.00\ndtype: object'), ([np.nan], '0   NaN\ndtype: float64'), ([None], '0    None\ndtype: object'), (['3.50', np.nan], '0    3.50\n1     NaN\ndtype: object'), ([3.5, np.nan], '0    3.5\n1    NaN\ndtype: float64'), ([3.5, np.nan, '3.50'], '0     3.5\n1     NaN\n2    3.50\ndtype: object'), ([3.5, None, '3.50'], '0     3.5\n1    None\n2    3.50\ndtype: object')])
    def test_repr_str_float_truncation(self, data, expected, using_infer_string):
        series = Series(data, dtype=object if '3.50' in data else None)
        result = repr(series)
        assert result == expected

    @pytest.mark.parametrize('float_format,expected', [('{:,.0f}'.format, '0   1,000\n1    test\ndtype: object'), ('{:.4f}'.format, '0   1000.0000\n1        test\ndtype: object')])
    def test_repr_float_format_in_object_col(self, float_format, expected):
        df = Series([1000.0, 'test'])
        with option_context('display.float_format', float_format):
            result = repr(df)
        assert result == expected

    def test_period(self):
        df = DataFrame({'A': pd.period_range('2013-01', periods=4, freq='M'), 'B': [pd.Period('2011-01', freq='M'), pd.Period('2011-02-01', freq='D'), pd.Period('2011-03-01 09:00', freq='h'), pd.Period('2011-04', freq='M')], 'C': list('abcd')})
        exp = '         A                 B  C\n0  2013-01           2011-01  a\n1  2013-02        2011-02-01  b\n2  2013-03  2011-03-01 09:00  c\n3  2013-04           2011-04  d'
        assert str(df) == exp

    @pytest.mark.parametrize('length, max_rows, min_rows, expected', [(10, 10, 10, 10), (10, 10, None, 10), (10, 8, None, 8), (20, 30, 10, 30), (50, 30, 10, 10), (100, 60, 10, 10), (60, 60, 10, 60), (61, 60, 10, 10)])
    def test_max_rows_fitted(self, length, min_rows, max_rows, expected):
        """Check that display logic is correct.

        GH #37359

        See description here:
        https://pandas.pydata.org/docs/dev/user_guide/options.html#frequently-used-options
        """
        formatter = fmt.DataFrameFormatter(DataFrame(np.random.default_rng(2).random((length, 3))), max_rows=max_rows, min_rows=min_rows)
        result = formatter.max_rows_fitted
        assert result == expected