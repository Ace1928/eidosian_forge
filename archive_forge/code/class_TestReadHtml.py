from collections.abc import Iterator
from functools import partial
from io import (
import os
from pathlib import Path
import re
import threading
from urllib.error import URLError
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.common import file_path_to_url
class TestReadHtml:

    def test_literal_html_deprecation(self, flavor_read_html):
        msg = "Passing literal html to 'read_html' is deprecated and will be removed in a future version. To read from a literal string, wrap it in a 'StringIO' object."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            flavor_read_html('<table>\n                <thead>\n                    <tr>\n                        <th>A</th>\n                        <th>B</th>\n                    </tr>\n                </thead>\n                <tbody>\n                    <tr>\n                        <td>1</td>\n                        <td>2</td>\n                    </tr>\n                </tbody>\n                <tbody>\n                    <tr>\n                        <td>3</td>\n                        <td>4</td>\n                    </tr>\n                </tbody>\n            </table>')

    @pytest.fixture
    def spam_data(self, datapath):
        return datapath('io', 'data', 'html', 'spam.html')

    @pytest.fixture
    def banklist_data(self, datapath):
        return datapath('io', 'data', 'html', 'banklist.html')

    def test_to_html_compat(self, flavor_read_html):
        df = DataFrame(np.random.default_rng(2).random((4, 3)), columns=pd.Index(list('abc'), dtype=object)).map('{:.3f}'.format).astype(float)
        out = df.to_html()
        res = flavor_read_html(StringIO(out), attrs={'class': 'dataframe'}, index_col=0)[0]
        tm.assert_frame_equal(res, df)

    def test_dtype_backend(self, string_storage, dtype_backend, flavor_read_html):
        df = DataFrame({'a': Series([1, np.nan, 3], dtype='Int64'), 'b': Series([1, 2, 3], dtype='Int64'), 'c': Series([1.5, np.nan, 2.5], dtype='Float64'), 'd': Series([1.5, 2.0, 2.5], dtype='Float64'), 'e': [True, False, None], 'f': [True, False, True], 'g': ['a', 'b', 'c'], 'h': ['a', 'b', None]})
        if string_storage == 'python':
            string_array = StringArray(np.array(['a', 'b', 'c'], dtype=np.object_))
            string_array_na = StringArray(np.array(['a', 'b', NA], dtype=np.object_))
        elif dtype_backend == 'pyarrow':
            pa = pytest.importorskip('pyarrow')
            from pandas.arrays import ArrowExtensionArray
            string_array = ArrowExtensionArray(pa.array(['a', 'b', 'c']))
            string_array_na = ArrowExtensionArray(pa.array(['a', 'b', None]))
        else:
            pa = pytest.importorskip('pyarrow')
            string_array = ArrowStringArray(pa.array(['a', 'b', 'c']))
            string_array_na = ArrowStringArray(pa.array(['a', 'b', None]))
        out = df.to_html(index=False)
        with pd.option_context('mode.string_storage', string_storage):
            result = flavor_read_html(StringIO(out), dtype_backend=dtype_backend)[0]
        expected = DataFrame({'a': Series([1, np.nan, 3], dtype='Int64'), 'b': Series([1, 2, 3], dtype='Int64'), 'c': Series([1.5, np.nan, 2.5], dtype='Float64'), 'd': Series([1.5, 2.0, 2.5], dtype='Float64'), 'e': Series([True, False, NA], dtype='boolean'), 'f': Series([True, False, True], dtype='boolean'), 'g': string_array, 'h': string_array_na})
        if dtype_backend == 'pyarrow':
            import pyarrow as pa
            from pandas.arrays import ArrowExtensionArray
            expected = DataFrame({col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True)) for col in expected.columns})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_banklist_url(self, httpserver, banklist_data, flavor_read_html):
        with open(banklist_data, encoding='utf-8') as f:
            httpserver.serve_content(content=f.read())
            df1 = flavor_read_html(httpserver.url, match='First Federal Bank of Florida')
            df2 = flavor_read_html(httpserver.url, match='Metcalf Bank')
        assert_framelist_equal(df1, df2)

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_spam_url(self, httpserver, spam_data, flavor_read_html):
        with open(spam_data, encoding='utf-8') as f:
            httpserver.serve_content(content=f.read())
            df1 = flavor_read_html(httpserver.url, match='.*Water.*')
            df2 = flavor_read_html(httpserver.url, match='Unit')
        assert_framelist_equal(df1, df2)

    @pytest.mark.slow
    def test_banklist(self, banklist_data, flavor_read_html):
        df1 = flavor_read_html(banklist_data, match='.*Florida.*', attrs={'id': 'table'})
        df2 = flavor_read_html(banklist_data, match='Metcalf Bank', attrs={'id': 'table'})
        assert_framelist_equal(df1, df2)

    def test_spam(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match='.*Water.*')
        df2 = flavor_read_html(spam_data, match='Unit')
        assert_framelist_equal(df1, df2)
        assert df1[0].iloc[0, 0] == 'Proximates'
        assert df1[0].columns[0] == 'Nutrient'

    def test_spam_no_match(self, spam_data, flavor_read_html):
        dfs = flavor_read_html(spam_data)
        for df in dfs:
            assert isinstance(df, DataFrame)

    def test_banklist_no_match(self, banklist_data, flavor_read_html):
        dfs = flavor_read_html(banklist_data, attrs={'id': 'table'})
        for df in dfs:
            assert isinstance(df, DataFrame)

    def test_spam_header(self, spam_data, flavor_read_html):
        df = flavor_read_html(spam_data, match='.*Water.*', header=2)[0]
        assert df.columns[0] == 'Proximates'
        assert not df.empty

    def test_skiprows_int(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match='.*Water.*', skiprows=1)
        df2 = flavor_read_html(spam_data, match='Unit', skiprows=1)
        assert_framelist_equal(df1, df2)

    def test_skiprows_range(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match='.*Water.*', skiprows=range(2))
        df2 = flavor_read_html(spam_data, match='Unit', skiprows=range(2))
        assert_framelist_equal(df1, df2)

    def test_skiprows_list(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match='.*Water.*', skiprows=[1, 2])
        df2 = flavor_read_html(spam_data, match='Unit', skiprows=[2, 1])
        assert_framelist_equal(df1, df2)

    def test_skiprows_set(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match='.*Water.*', skiprows={1, 2})
        df2 = flavor_read_html(spam_data, match='Unit', skiprows={2, 1})
        assert_framelist_equal(df1, df2)

    def test_skiprows_slice(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match='.*Water.*', skiprows=1)
        df2 = flavor_read_html(spam_data, match='Unit', skiprows=1)
        assert_framelist_equal(df1, df2)

    def test_skiprows_slice_short(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match='.*Water.*', skiprows=slice(2))
        df2 = flavor_read_html(spam_data, match='Unit', skiprows=slice(2))
        assert_framelist_equal(df1, df2)

    def test_skiprows_slice_long(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match='.*Water.*', skiprows=slice(2, 5))
        df2 = flavor_read_html(spam_data, match='Unit', skiprows=slice(4, 1, -1))
        assert_framelist_equal(df1, df2)

    def test_skiprows_ndarray(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match='.*Water.*', skiprows=np.arange(2))
        df2 = flavor_read_html(spam_data, match='Unit', skiprows=np.arange(2))
        assert_framelist_equal(df1, df2)

    def test_skiprows_invalid(self, spam_data, flavor_read_html):
        with pytest.raises(TypeError, match='is not a valid type for skipping rows'):
            flavor_read_html(spam_data, match='.*Water.*', skiprows='asdf')

    def test_index(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match='.*Water.*', index_col=0)
        df2 = flavor_read_html(spam_data, match='Unit', index_col=0)
        assert_framelist_equal(df1, df2)

    def test_header_and_index_no_types(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match='.*Water.*', header=1, index_col=0)
        df2 = flavor_read_html(spam_data, match='Unit', header=1, index_col=0)
        assert_framelist_equal(df1, df2)

    def test_header_and_index_with_types(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match='.*Water.*', header=1, index_col=0)
        df2 = flavor_read_html(spam_data, match='Unit', header=1, index_col=0)
        assert_framelist_equal(df1, df2)

    def test_infer_types(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match='.*Water.*', index_col=0)
        df2 = flavor_read_html(spam_data, match='Unit', index_col=0)
        assert_framelist_equal(df1, df2)

    def test_string_io(self, spam_data, flavor_read_html):
        with open(spam_data, encoding='UTF-8') as f:
            data1 = StringIO(f.read())
        with open(spam_data, encoding='UTF-8') as f:
            data2 = StringIO(f.read())
        df1 = flavor_read_html(data1, match='.*Water.*')
        df2 = flavor_read_html(data2, match='Unit')
        assert_framelist_equal(df1, df2)

    def test_string(self, spam_data, flavor_read_html):
        with open(spam_data, encoding='UTF-8') as f:
            data = f.read()
        df1 = flavor_read_html(StringIO(data), match='.*Water.*')
        df2 = flavor_read_html(StringIO(data), match='Unit')
        assert_framelist_equal(df1, df2)

    def test_file_like(self, spam_data, flavor_read_html):
        with open(spam_data, encoding='UTF-8') as f:
            df1 = flavor_read_html(f, match='.*Water.*')
        with open(spam_data, encoding='UTF-8') as f:
            df2 = flavor_read_html(f, match='Unit')
        assert_framelist_equal(df1, df2)

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_bad_url_protocol(self, httpserver, flavor_read_html):
        httpserver.serve_content('urlopen error unknown url type: git', code=404)
        with pytest.raises(URLError, match='urlopen error unknown url type: git'):
            flavor_read_html('git://github.com', match='.*Water.*')

    @pytest.mark.slow
    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_invalid_url(self, httpserver, flavor_read_html):
        httpserver.serve_content('Name or service not known', code=404)
        with pytest.raises((URLError, ValueError), match='HTTP Error 404: NOT FOUND'):
            flavor_read_html(httpserver.url, match='.*Water.*')

    @pytest.mark.slow
    def test_file_url(self, banklist_data, flavor_read_html):
        url = banklist_data
        dfs = flavor_read_html(file_path_to_url(os.path.abspath(url)), match='First', attrs={'id': 'table'})
        assert isinstance(dfs, list)
        for df in dfs:
            assert isinstance(df, DataFrame)

    @pytest.mark.slow
    def test_invalid_table_attrs(self, banklist_data, flavor_read_html):
        url = banklist_data
        with pytest.raises(ValueError, match='No tables found'):
            flavor_read_html(url, match='First Federal Bank of Florida', attrs={'id': 'tasdfable'})

    @pytest.mark.slow
    def test_multiindex_header(self, banklist_data, flavor_read_html):
        df = flavor_read_html(banklist_data, match='Metcalf', attrs={'id': 'table'}, header=[0, 1])[0]
        assert isinstance(df.columns, MultiIndex)

    @pytest.mark.slow
    def test_multiindex_index(self, banklist_data, flavor_read_html):
        df = flavor_read_html(banklist_data, match='Metcalf', attrs={'id': 'table'}, index_col=[0, 1])[0]
        assert isinstance(df.index, MultiIndex)

    @pytest.mark.slow
    def test_multiindex_header_index(self, banklist_data, flavor_read_html):
        df = flavor_read_html(banklist_data, match='Metcalf', attrs={'id': 'table'}, header=[0, 1], index_col=[0, 1])[0]
        assert isinstance(df.columns, MultiIndex)
        assert isinstance(df.index, MultiIndex)

    @pytest.mark.slow
    def test_multiindex_header_skiprows_tuples(self, banklist_data, flavor_read_html):
        df = flavor_read_html(banklist_data, match='Metcalf', attrs={'id': 'table'}, header=[0, 1], skiprows=1)[0]
        assert isinstance(df.columns, MultiIndex)

    @pytest.mark.slow
    def test_multiindex_header_skiprows(self, banklist_data, flavor_read_html):
        df = flavor_read_html(banklist_data, match='Metcalf', attrs={'id': 'table'}, header=[0, 1], skiprows=1)[0]
        assert isinstance(df.columns, MultiIndex)

    @pytest.mark.slow
    def test_multiindex_header_index_skiprows(self, banklist_data, flavor_read_html):
        df = flavor_read_html(banklist_data, match='Metcalf', attrs={'id': 'table'}, header=[0, 1], index_col=[0, 1], skiprows=1)[0]
        assert isinstance(df.index, MultiIndex)
        assert isinstance(df.columns, MultiIndex)

    @pytest.mark.slow
    def test_regex_idempotency(self, banklist_data, flavor_read_html):
        url = banklist_data
        dfs = flavor_read_html(file_path_to_url(os.path.abspath(url)), match=re.compile(re.compile('Florida')), attrs={'id': 'table'})
        assert isinstance(dfs, list)
        for df in dfs:
            assert isinstance(df, DataFrame)

    def test_negative_skiprows(self, spam_data, flavor_read_html):
        msg = '\\(you passed a negative value\\)'
        with pytest.raises(ValueError, match=msg):
            flavor_read_html(spam_data, match='Water', skiprows=-1)

    @pytest.fixture
    def python_docs(self):
        return '\n          <table class="contentstable" align="center"><tr>\n            <td width="50%">\n            <p class="biglink"><a class="biglink" href="whatsnew/2.7.html">What\'s new in Python 2.7?</a><br/>\n                <span class="linkdescr">or <a href="whatsnew/index.html">all "What\'s new" documents</a> since 2.0</span></p>\n            <p class="biglink"><a class="biglink" href="tutorial/index.html">Tutorial</a><br/>\n                <span class="linkdescr">start here</span></p>\n            <p class="biglink"><a class="biglink" href="library/index.html">Library Reference</a><br/>\n                <span class="linkdescr">keep this under your pillow</span></p>\n            <p class="biglink"><a class="biglink" href="reference/index.html">Language Reference</a><br/>\n                <span class="linkdescr">describes syntax and language elements</span></p>\n            <p class="biglink"><a class="biglink" href="using/index.html">Python Setup and Usage</a><br/>\n                <span class="linkdescr">how to use Python on different platforms</span></p>\n            <p class="biglink"><a class="biglink" href="howto/index.html">Python HOWTOs</a><br/>\n                <span class="linkdescr">in-depth documents on specific topics</span></p>\n            </td><td width="50%">\n            <p class="biglink"><a class="biglink" href="installing/index.html">Installing Python Modules</a><br/>\n                <span class="linkdescr">installing from the Python Package Index &amp; other sources</span></p>\n            <p class="biglink"><a class="biglink" href="distributing/index.html">Distributing Python Modules</a><br/>\n                <span class="linkdescr">publishing modules for installation by others</span></p>\n            <p class="biglink"><a class="biglink" href="extending/index.html">Extending and Embedding</a><br/>\n                <span class="linkdescr">tutorial for C/C++ programmers</span></p>\n            <p class="biglink"><a class="biglink" href="c-api/index.html">Python/C API</a><br/>\n                <span class="linkdescr">reference for C/C++ programmers</span></p>\n            <p class="biglink"><a class="biglink" href="faq/index.html">FAQs</a><br/>\n                <span class="linkdescr">frequently asked questions (with answers!)</span></p>\n            </td></tr>\n        </table>\n\n        <p><strong>Indices and tables:</strong></p>\n        <table class="contentstable" align="center"><tr>\n            <td width="50%">\n            <p class="biglink"><a class="biglink" href="py-modindex.html">Python Global Module Index</a><br/>\n                <span class="linkdescr">quick access to all modules</span></p>\n            <p class="biglink"><a class="biglink" href="genindex.html">General Index</a><br/>\n                <span class="linkdescr">all functions, classes, terms</span></p>\n            <p class="biglink"><a class="biglink" href="glossary.html">Glossary</a><br/>\n                <span class="linkdescr">the most important terms explained</span></p>\n            </td><td width="50%">\n            <p class="biglink"><a class="biglink" href="search.html">Search page</a><br/>\n                <span class="linkdescr">search this documentation</span></p>\n            <p class="biglink"><a class="biglink" href="contents.html">Complete Table of Contents</a><br/>\n                <span class="linkdescr">lists all sections and subsections</span></p>\n            </td></tr>\n        </table>\n        '

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_multiple_matches(self, python_docs, httpserver, flavor_read_html):
        httpserver.serve_content(content=python_docs)
        dfs = flavor_read_html(httpserver.url, match='Python')
        assert len(dfs) > 1

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_python_docs_table(self, python_docs, httpserver, flavor_read_html):
        httpserver.serve_content(content=python_docs)
        dfs = flavor_read_html(httpserver.url, match='Python')
        zz = [df.iloc[0, 0][0:4] for df in dfs]
        assert sorted(zz) == ['Pyth', 'What']

    def test_empty_tables(self, flavor_read_html):
        """
        Make sure that read_html ignores empty tables.
        """
        html = '\n            <table>\n                <thead>\n                    <tr>\n                        <th>A</th>\n                        <th>B</th>\n                    </tr>\n                </thead>\n                <tbody>\n                    <tr>\n                        <td>1</td>\n                        <td>2</td>\n                    </tr>\n                </tbody>\n            </table>\n            <table>\n                <tbody>\n                </tbody>\n            </table>\n        '
        result = flavor_read_html(StringIO(html))
        assert len(result) == 1

    def test_multiple_tbody(self, flavor_read_html):
        result = flavor_read_html(StringIO('<table>\n            <thead>\n                <tr>\n                    <th>A</th>\n                    <th>B</th>\n                </tr>\n            </thead>\n            <tbody>\n                <tr>\n                    <td>1</td>\n                    <td>2</td>\n                </tr>\n            </tbody>\n            <tbody>\n                <tr>\n                    <td>3</td>\n                    <td>4</td>\n                </tr>\n            </tbody>\n        </table>'))[0]
        expected = DataFrame(data=[[1, 2], [3, 4]], columns=['A', 'B'])
        tm.assert_frame_equal(result, expected)

    def test_header_and_one_column(self, flavor_read_html):
        """
        Don't fail with bs4 when there is a header and only one column
        as described in issue #9178
        """
        result = flavor_read_html(StringIO('<table>\n                <thead>\n                    <tr>\n                        <th>Header</th>\n                    </tr>\n                </thead>\n                <tbody>\n                    <tr>\n                        <td>first</td>\n                    </tr>\n                </tbody>\n            </table>'))[0]
        expected = DataFrame(data={'Header': 'first'}, index=[0])
        tm.assert_frame_equal(result, expected)

    def test_thead_without_tr(self, flavor_read_html):
        """
        Ensure parser adds <tr> within <thead> on malformed HTML.
        """
        result = flavor_read_html(StringIO('<table>\n            <thead>\n                <tr>\n                    <th>Country</th>\n                    <th>Municipality</th>\n                    <th>Year</th>\n                </tr>\n            </thead>\n            <tbody>\n                <tr>\n                    <td>Ukraine</td>\n                    <th>Odessa</th>\n                    <td>1944</td>\n                </tr>\n            </tbody>\n        </table>'))[0]
        expected = DataFrame(data=[['Ukraine', 'Odessa', 1944]], columns=['Country', 'Municipality', 'Year'])
        tm.assert_frame_equal(result, expected)

    def test_tfoot_read(self, flavor_read_html):
        """
        Make sure that read_html reads tfoot, containing td or th.
        Ignores empty tfoot
        """
        data_template = '<table>\n            <thead>\n                <tr>\n                    <th>A</th>\n                    <th>B</th>\n                </tr>\n            </thead>\n            <tbody>\n                <tr>\n                    <td>bodyA</td>\n                    <td>bodyB</td>\n                </tr>\n            </tbody>\n            <tfoot>\n                {footer}\n            </tfoot>\n        </table>'
        expected1 = DataFrame(data=[['bodyA', 'bodyB']], columns=['A', 'B'])
        expected2 = DataFrame(data=[['bodyA', 'bodyB'], ['footA', 'footB']], columns=['A', 'B'])
        data1 = data_template.format(footer='')
        data2 = data_template.format(footer='<tr><td>footA</td><th>footB</th></tr>')
        result1 = flavor_read_html(StringIO(data1))[0]
        result2 = flavor_read_html(StringIO(data2))[0]
        tm.assert_frame_equal(result1, expected1)
        tm.assert_frame_equal(result2, expected2)

    def test_parse_header_of_non_string_column(self, flavor_read_html):
        result = flavor_read_html(StringIO('\n            <table>\n                <tr>\n                    <td>S</td>\n                    <td>I</td>\n                </tr>\n                <tr>\n                    <td>text</td>\n                    <td>1944</td>\n                </tr>\n            </table>\n        '), header=0)[0]
        expected = DataFrame([['text', 1944]], columns=('S', 'I'))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.slow
    def test_banklist_header(self, banklist_data, datapath, flavor_read_html):
        from pandas.io.html import _remove_whitespace

        def try_remove_ws(x):
            try:
                return _remove_whitespace(x)
            except AttributeError:
                return x
        df = flavor_read_html(banklist_data, match='Metcalf', attrs={'id': 'table'})[0]
        ground_truth = read_csv(datapath('io', 'data', 'csv', 'banklist.csv'), converters={'Updated Date': Timestamp, 'Closing Date': Timestamp})
        assert df.shape == ground_truth.shape
        old = ['First Vietnamese American Bank In Vietnamese', 'Westernbank Puerto Rico En Espanol', 'R-G Premier Bank of Puerto Rico En Espanol', 'Eurobank En Espanol', 'Sanderson State Bank En Espanol', 'Washington Mutual Bank (Including its subsidiary Washington Mutual Bank FSB)', 'Silver State Bank En Espanol', 'AmTrade International Bank En Espanol', 'Hamilton Bank, NA En Espanol', 'The Citizens Savings Bank Pioneer Community Bank, Inc.']
        new = ['First Vietnamese American Bank', 'Westernbank Puerto Rico', 'R-G Premier Bank of Puerto Rico', 'Eurobank', 'Sanderson State Bank', 'Washington Mutual Bank', 'Silver State Bank', 'AmTrade International Bank', 'Hamilton Bank, NA', 'The Citizens Savings Bank']
        dfnew = df.map(try_remove_ws).replace(old, new)
        gtnew = ground_truth.map(try_remove_ws)
        converted = dfnew
        date_cols = ['Closing Date', 'Updated Date']
        converted[date_cols] = converted[date_cols].apply(to_datetime)
        tm.assert_frame_equal(converted, gtnew)

    @pytest.mark.slow
    def test_gold_canyon(self, banklist_data, flavor_read_html):
        gc = 'Gold Canyon'
        with open(banklist_data, encoding='utf-8') as f:
            raw_text = f.read()
        assert gc in raw_text
        df = flavor_read_html(banklist_data, match='Gold Canyon', attrs={'id': 'table'})[0]
        assert gc in df.to_string()

    def test_different_number_of_cols(self, flavor_read_html):
        expected = flavor_read_html(StringIO('<table>\n                        <thead>\n                            <tr style="text-align: right;">\n                            <th></th>\n                            <th>C_l0_g0</th>\n                            <th>C_l0_g1</th>\n                            <th>C_l0_g2</th>\n                            <th>C_l0_g3</th>\n                            <th>C_l0_g4</th>\n                            </tr>\n                        </thead>\n                        <tbody>\n                            <tr>\n                            <th>R_l0_g0</th>\n                            <td> 0.763</td>\n                            <td> 0.233</td>\n                            <td> nan</td>\n                            <td> nan</td>\n                            <td> nan</td>\n                            </tr>\n                            <tr>\n                            <th>R_l0_g1</th>\n                            <td> 0.244</td>\n                            <td> 0.285</td>\n                            <td> 0.392</td>\n                            <td> 0.137</td>\n                            <td> 0.222</td>\n                            </tr>\n                        </tbody>\n                    </table>'), index_col=0)[0]
        result = flavor_read_html(StringIO('<table>\n                    <thead>\n                        <tr style="text-align: right;">\n                        <th></th>\n                        <th>C_l0_g0</th>\n                        <th>C_l0_g1</th>\n                        <th>C_l0_g2</th>\n                        <th>C_l0_g3</th>\n                        <th>C_l0_g4</th>\n                        </tr>\n                    </thead>\n                    <tbody>\n                        <tr>\n                        <th>R_l0_g0</th>\n                        <td> 0.763</td>\n                        <td> 0.233</td>\n                        </tr>\n                        <tr>\n                        <th>R_l0_g1</th>\n                        <td> 0.244</td>\n                        <td> 0.285</td>\n                        <td> 0.392</td>\n                        <td> 0.137</td>\n                        <td> 0.222</td>\n                        </tr>\n                    </tbody>\n                 </table>'), index_col=0)[0]
        tm.assert_frame_equal(result, expected)

    def test_colspan_rowspan_1(self, flavor_read_html):
        result = flavor_read_html(StringIO('\n            <table>\n                <tr>\n                    <th>A</th>\n                    <th colspan="1">B</th>\n                    <th rowspan="1">C</th>\n                </tr>\n                <tr>\n                    <td>a</td>\n                    <td>b</td>\n                    <td>c</td>\n                </tr>\n            </table>\n        '))[0]
        expected = DataFrame([['a', 'b', 'c']], columns=['A', 'B', 'C'])
        tm.assert_frame_equal(result, expected)

    def test_colspan_rowspan_copy_values(self, flavor_read_html):
        result = flavor_read_html(StringIO('\n            <table>\n                <tr>\n                    <td colspan="2">X</td>\n                    <td>Y</td>\n                    <td rowspan="2">Z</td>\n                    <td>W</td>\n                </tr>\n                <tr>\n                    <td>A</td>\n                    <td colspan="2">B</td>\n                    <td>C</td>\n                </tr>\n            </table>\n        '), header=0)[0]
        expected = DataFrame(data=[['A', 'B', 'B', 'Z', 'C']], columns=['X', 'X.1', 'Y', 'Z', 'W'])
        tm.assert_frame_equal(result, expected)

    def test_colspan_rowspan_both_not_1(self, flavor_read_html):
        result = flavor_read_html(StringIO('\n            <table>\n                <tr>\n                    <td rowspan="2">A</td>\n                    <td rowspan="2" colspan="3">B</td>\n                    <td>C</td>\n                </tr>\n                <tr>\n                    <td>D</td>\n                </tr>\n            </table>\n        '), header=0)[0]
        expected = DataFrame(data=[['A', 'B', 'B', 'B', 'D']], columns=['A', 'B', 'B.1', 'B.2', 'C'])
        tm.assert_frame_equal(result, expected)

    def test_rowspan_at_end_of_row(self, flavor_read_html):
        result = flavor_read_html(StringIO('\n            <table>\n                <tr>\n                    <td>A</td>\n                    <td rowspan="2">B</td>\n                </tr>\n                <tr>\n                    <td>C</td>\n                </tr>\n            </table>\n        '), header=0)[0]
        expected = DataFrame(data=[['C', 'B']], columns=['A', 'B'])
        tm.assert_frame_equal(result, expected)

    def test_rowspan_only_rows(self, flavor_read_html):
        result = flavor_read_html(StringIO('\n            <table>\n                <tr>\n                    <td rowspan="3">A</td>\n                    <td rowspan="3">B</td>\n                </tr>\n            </table>\n        '), header=0)[0]
        expected = DataFrame(data=[['A', 'B'], ['A', 'B']], columns=['A', 'B'])
        tm.assert_frame_equal(result, expected)

    def test_header_inferred_from_rows_with_only_th(self, flavor_read_html):
        result = flavor_read_html(StringIO('\n            <table>\n                <tr>\n                    <th>A</th>\n                    <th>B</th>\n                </tr>\n                <tr>\n                    <th>a</th>\n                    <th>b</th>\n                </tr>\n                <tr>\n                    <td>1</td>\n                    <td>2</td>\n                </tr>\n            </table>\n        '))[0]
        columns = MultiIndex(levels=[['A', 'B'], ['a', 'b']], codes=[[0, 1], [0, 1]])
        expected = DataFrame(data=[[1, 2]], columns=columns)
        tm.assert_frame_equal(result, expected)

    def test_parse_dates_list(self, flavor_read_html):
        df = DataFrame({'date': date_range('1/1/2001', periods=10)})
        expected = df.to_html()
        res = flavor_read_html(StringIO(expected), parse_dates=[1], index_col=0)
        tm.assert_frame_equal(df, res[0])
        res = flavor_read_html(StringIO(expected), parse_dates=['date'], index_col=0)
        tm.assert_frame_equal(df, res[0])

    def test_parse_dates_combine(self, flavor_read_html):
        raw_dates = Series(date_range('1/1/2001', periods=10))
        df = DataFrame({'date': raw_dates.map(lambda x: str(x.date())), 'time': raw_dates.map(lambda x: str(x.time()))})
        res = flavor_read_html(StringIO(df.to_html()), parse_dates={'datetime': [1, 2]}, index_col=1)
        newdf = DataFrame({'datetime': raw_dates})
        tm.assert_frame_equal(newdf, res[0])

    def test_wikipedia_states_table(self, datapath, flavor_read_html):
        data = datapath('io', 'data', 'html', 'wikipedia_states.html')
        assert os.path.isfile(data), f'{repr(data)} is not a file'
        assert os.path.getsize(data), f'{repr(data)} is an empty file'
        result = flavor_read_html(data, match='Arizona', header=1)[0]
        assert result.shape == (60, 12)
        assert 'Unnamed' in result.columns[-1]
        assert result['sq mi'].dtype == np.dtype('float64')
        assert np.allclose(result.loc[0, 'sq mi'], 665384.04)

    def test_wikipedia_states_multiindex(self, datapath, flavor_read_html):
        data = datapath('io', 'data', 'html', 'wikipedia_states.html')
        result = flavor_read_html(data, match='Arizona', index_col=0)[0]
        assert result.shape == (60, 11)
        assert 'Unnamed' in result.columns[-1][1]
        assert result.columns.nlevels == 2
        assert np.allclose(result.loc['Alaska', ('Total area[2]', 'sq mi')], 665384.04)

    def test_parser_error_on_empty_header_row(self, flavor_read_html):
        result = flavor_read_html(StringIO('\n                <table>\n                    <thead>\n                        <tr><th></th><th></tr>\n                        <tr><th>A</th><th>B</th></tr>\n                    </thead>\n                    <tbody>\n                        <tr><td>a</td><td>b</td></tr>\n                    </tbody>\n                </table>\n            '), header=[0, 1])
        expected = DataFrame([['a', 'b']], columns=MultiIndex.from_tuples([('Unnamed: 0_level_0', 'A'), ('Unnamed: 1_level_0', 'B')]))
        tm.assert_frame_equal(result[0], expected)

    def test_decimal_rows(self, flavor_read_html):
        result = flavor_read_html(StringIO('<html>\n            <body>\n             <table>\n                <thead>\n                    <tr>\n                        <th>Header</th>\n                    </tr>\n                </thead>\n                <tbody>\n                    <tr>\n                        <td>1100#101</td>\n                    </tr>\n                </tbody>\n            </table>\n            </body>\n        </html>'), decimal='#')[0]
        expected = DataFrame(data={'Header': 1100.101}, index=[0])
        assert result['Header'].dtype == np.dtype('float64')
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('arg', [True, False])
    def test_bool_header_arg(self, spam_data, arg, flavor_read_html):
        msg = re.escape('Passing a bool to header is invalid. Use header=None for no header or header=int or list-like of ints to specify the row(s) making up the column names')
        with pytest.raises(TypeError, match=msg):
            flavor_read_html(spam_data, header=arg)

    def test_converters(self, flavor_read_html):
        result = flavor_read_html(StringIO('<table>\n                 <thead>\n                   <tr>\n                     <th>a</th>\n                    </tr>\n                 </thead>\n                 <tbody>\n                   <tr>\n                     <td> 0.763</td>\n                   </tr>\n                   <tr>\n                     <td> 0.244</td>\n                   </tr>\n                 </tbody>\n               </table>'), converters={'a': str})[0]
        expected = DataFrame({'a': ['0.763', '0.244']})
        tm.assert_frame_equal(result, expected)

    def test_na_values(self, flavor_read_html):
        result = flavor_read_html(StringIO('<table>\n                 <thead>\n                   <tr>\n                     <th>a</th>\n                   </tr>\n                 </thead>\n                 <tbody>\n                   <tr>\n                     <td> 0.763</td>\n                   </tr>\n                   <tr>\n                     <td> 0.244</td>\n                   </tr>\n                 </tbody>\n               </table>'), na_values=[0.244])[0]
        expected = DataFrame({'a': [0.763, np.nan]})
        tm.assert_frame_equal(result, expected)

    def test_keep_default_na(self, flavor_read_html):
        html_data = '<table>\n                        <thead>\n                            <tr>\n                            <th>a</th>\n                            </tr>\n                        </thead>\n                        <tbody>\n                            <tr>\n                            <td> N/A</td>\n                            </tr>\n                            <tr>\n                            <td> NA</td>\n                            </tr>\n                        </tbody>\n                    </table>'
        expected_df = DataFrame({'a': ['N/A', 'NA']})
        html_df = flavor_read_html(StringIO(html_data), keep_default_na=False)[0]
        tm.assert_frame_equal(expected_df, html_df)
        expected_df = DataFrame({'a': [np.nan, np.nan]})
        html_df = flavor_read_html(StringIO(html_data), keep_default_na=True)[0]
        tm.assert_frame_equal(expected_df, html_df)

    def test_preserve_empty_rows(self, flavor_read_html):
        result = flavor_read_html(StringIO('\n            <table>\n                <tr>\n                    <th>A</th>\n                    <th>B</th>\n                </tr>\n                <tr>\n                    <td>a</td>\n                    <td>b</td>\n                </tr>\n                <tr>\n                    <td></td>\n                    <td></td>\n                </tr>\n            </table>\n        '))[0]
        expected = DataFrame(data=[['a', 'b'], [np.nan, np.nan]], columns=['A', 'B'])
        tm.assert_frame_equal(result, expected)

    def test_ignore_empty_rows_when_inferring_header(self, flavor_read_html):
        result = flavor_read_html(StringIO('\n            <table>\n                <thead>\n                    <tr><th></th><th></tr>\n                    <tr><th>A</th><th>B</th></tr>\n                    <tr><th>a</th><th>b</th></tr>\n                </thead>\n                <tbody>\n                    <tr><td>1</td><td>2</td></tr>\n                </tbody>\n            </table>\n        '))[0]
        columns = MultiIndex(levels=[['A', 'B'], ['a', 'b']], codes=[[0, 1], [0, 1]])
        expected = DataFrame(data=[[1, 2]], columns=columns)
        tm.assert_frame_equal(result, expected)

    def test_multiple_header_rows(self, flavor_read_html):
        expected_df = DataFrame(data=[('Hillary', 68, 'D'), ('Bernie', 74, 'D'), ('Donald', 69, 'R')])
        expected_df.columns = [['Unnamed: 0_level_0', 'Age', 'Party'], ['Name', 'Unnamed: 1_level_1', 'Unnamed: 2_level_1']]
        html = expected_df.to_html(index=False)
        html_df = flavor_read_html(StringIO(html))[0]
        tm.assert_frame_equal(expected_df, html_df)

    def test_works_on_valid_markup(self, datapath, flavor_read_html):
        filename = datapath('io', 'data', 'html', 'valid_markup.html')
        dfs = flavor_read_html(filename, index_col=0)
        assert isinstance(dfs, list)
        assert isinstance(dfs[0], DataFrame)

    @pytest.mark.slow
    def test_fallback_success(self, datapath, flavor_read_html):
        banklist_data = datapath('io', 'data', 'html', 'banklist.html')
        flavor_read_html(banklist_data, match='.*Water.*', flavor=['lxml', 'html5lib'])

    def test_to_html_timestamp(self):
        rng = date_range('2000-01-01', periods=10)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), index=rng)
        result = df.to_html()
        assert '2000-01-01' in result

    def test_to_html_borderless(self):
        df = DataFrame([{'A': 1, 'B': 2}])
        out_border_default = df.to_html()
        out_border_true = df.to_html(border=True)
        out_border_explicit_default = df.to_html(border=1)
        out_border_nondefault = df.to_html(border=2)
        out_border_zero = df.to_html(border=0)
        out_border_false = df.to_html(border=False)
        assert ' border="1"' in out_border_default
        assert out_border_true == out_border_default
        assert out_border_default == out_border_explicit_default
        assert out_border_default != out_border_nondefault
        assert ' border="2"' in out_border_nondefault
        assert ' border="0"' not in out_border_zero
        assert ' border' not in out_border_false
        assert out_border_zero == out_border_false

    @pytest.mark.parametrize('displayed_only,exp0,exp1', [(True, DataFrame(['foo']), None), (False, DataFrame(['foo  bar  baz  qux']), DataFrame(['foo']))])
    def test_displayed_only(self, displayed_only, exp0, exp1, flavor_read_html):
        data = '<html>\n          <body>\n            <table>\n              <tr>\n                <td>\n                  foo\n                  <span style="display:none;text-align:center">bar</span>\n                  <span style="display:none">baz</span>\n                  <span style="display: none">qux</span>\n                </td>\n              </tr>\n            </table>\n            <table style="display: none">\n              <tr>\n                <td>foo</td>\n              </tr>\n            </table>\n          </body>\n        </html>'
        dfs = flavor_read_html(StringIO(data), displayed_only=displayed_only)
        tm.assert_frame_equal(dfs[0], exp0)
        if exp1 is not None:
            tm.assert_frame_equal(dfs[1], exp1)
        else:
            assert len(dfs) == 1

    @pytest.mark.parametrize('displayed_only', [True, False])
    def test_displayed_only_with_many_elements(self, displayed_only, flavor_read_html):
        html_table = '\n        <table>\n            <tr>\n                <th>A</th>\n                <th>B</th>\n            </tr>\n            <tr>\n                <td>1</td>\n                <td>2</td>\n            </tr>\n            <tr>\n                <td><span style="display:none"></span>4</td>\n                <td>5</td>\n            </tr>\n        </table>\n        '
        result = flavor_read_html(StringIO(html_table), displayed_only=displayed_only)[0]
        expected = DataFrame({'A': [1, 4], 'B': [2, 5]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:You provided Unicode markup but also provided a value for from_encoding.*:UserWarning')
    def test_encode(self, html_encoding_file, flavor_read_html):
        base_path = os.path.basename(html_encoding_file)
        root = os.path.splitext(base_path)[0]
        _, encoding = root.split('_')
        try:
            with open(html_encoding_file, 'rb') as fobj:
                from_string = flavor_read_html(fobj.read(), encoding=encoding, index_col=0).pop()
            with open(html_encoding_file, 'rb') as fobj:
                from_file_like = flavor_read_html(BytesIO(fobj.read()), encoding=encoding, index_col=0).pop()
            from_filename = flavor_read_html(html_encoding_file, encoding=encoding, index_col=0).pop()
            tm.assert_frame_equal(from_string, from_file_like)
            tm.assert_frame_equal(from_string, from_filename)
        except Exception:
            if is_platform_windows():
                if '16' in encoding or '32' in encoding:
                    pytest.skip()
            raise

    def test_parse_failure_unseekable(self, flavor_read_html):
        if flavor_read_html.keywords.get('flavor') == 'lxml':
            pytest.skip('Not applicable for lxml')

        class UnseekableStringIO(StringIO):

            def seekable(self):
                return False
        bad = UnseekableStringIO('\n            <table><tr><td>spam<foobr />eggs</td></tr></table>')
        assert flavor_read_html(bad)
        with pytest.raises(ValueError, match='passed a non-rewindable file object'):
            flavor_read_html(bad)

    def test_parse_failure_rewinds(self, flavor_read_html):

        class MockFile:

            def __init__(self, data) -> None:
                self.data = data
                self.at_end = False

            def read(self, size=None):
                data = '' if self.at_end else self.data
                self.at_end = True
                return data

            def seek(self, offset):
                self.at_end = False

            def seekable(self):
                return True

            def __next__(self):
                ...

            def __iter__(self) -> Iterator:
                return self
        good = MockFile('<table><tr><td>spam<br />eggs</td></tr></table>')
        bad = MockFile('<table><tr><td>spam<foobr />eggs</td></tr></table>')
        assert flavor_read_html(good)
        assert flavor_read_html(bad)

    @pytest.mark.slow
    @pytest.mark.single_cpu
    def test_importcheck_thread_safety(self, datapath, flavor_read_html):

        class ErrorThread(threading.Thread):

            def run(self):
                try:
                    super().run()
                except Exception as err:
                    self.err = err
                else:
                    self.err = None
        filename = datapath('io', 'data', 'html', 'valid_markup.html')
        helper_thread1 = ErrorThread(target=flavor_read_html, args=(filename,))
        helper_thread2 = ErrorThread(target=flavor_read_html, args=(filename,))
        helper_thread1.start()
        helper_thread2.start()
        while helper_thread1.is_alive() or helper_thread2.is_alive():
            pass
        assert None is helper_thread1.err is helper_thread2.err

    def test_parse_path_object(self, datapath, flavor_read_html):
        file_path_string = datapath('io', 'data', 'html', 'spam.html')
        file_path = Path(file_path_string)
        df1 = flavor_read_html(file_path_string)[0]
        df2 = flavor_read_html(file_path)[0]
        tm.assert_frame_equal(df1, df2)

    def test_parse_br_as_space(self, flavor_read_html):
        result = flavor_read_html(StringIO('\n            <table>\n                <tr>\n                    <th>A</th>\n                </tr>\n                <tr>\n                    <td>word1<br>word2</td>\n                </tr>\n            </table>\n        '))[0]
        expected = DataFrame(data=[['word1 word2']], columns=['A'])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('arg', ['all', 'body', 'header', 'footer'])
    def test_extract_links(self, arg, flavor_read_html):
        gh_13141_data = '\n          <table>\n            <tr>\n              <th>HTTP</th>\n              <th>FTP</th>\n              <th><a href="https://en.wiktionary.org/wiki/linkless">Linkless</a></th>\n            </tr>\n            <tr>\n              <td><a href="https://en.wikipedia.org/">Wikipedia</a></td>\n              <td>SURROUNDING <a href="ftp://ftp.us.debian.org/">Debian</a> TEXT</td>\n              <td>Linkless</td>\n            </tr>\n            <tfoot>\n              <tr>\n                <td><a href="https://en.wikipedia.org/wiki/Page_footer">Footer</a></td>\n                <td>\n                  Multiple <a href="1">links:</a> <a href="2">Only first captured.</a>\n                </td>\n              </tr>\n            </tfoot>\n          </table>\n          '
        gh_13141_expected = {'head_ignore': ['HTTP', 'FTP', 'Linkless'], 'head_extract': [('HTTP', None), ('FTP', None), ('Linkless', 'https://en.wiktionary.org/wiki/linkless')], 'body_ignore': ['Wikipedia', 'SURROUNDING Debian TEXT', 'Linkless'], 'body_extract': [('Wikipedia', 'https://en.wikipedia.org/'), ('SURROUNDING Debian TEXT', 'ftp://ftp.us.debian.org/'), ('Linkless', None)], 'footer_ignore': ['Footer', 'Multiple links: Only first captured.', None], 'footer_extract': [('Footer', 'https://en.wikipedia.org/wiki/Page_footer'), ('Multiple links: Only first captured.', '1'), None]}
        data_exp = gh_13141_expected['body_ignore']
        foot_exp = gh_13141_expected['footer_ignore']
        head_exp = gh_13141_expected['head_ignore']
        if arg == 'all':
            data_exp = gh_13141_expected['body_extract']
            foot_exp = gh_13141_expected['footer_extract']
            head_exp = gh_13141_expected['head_extract']
        elif arg == 'body':
            data_exp = gh_13141_expected['body_extract']
        elif arg == 'footer':
            foot_exp = gh_13141_expected['footer_extract']
        elif arg == 'header':
            head_exp = gh_13141_expected['head_extract']
        result = flavor_read_html(StringIO(gh_13141_data), extract_links=arg)[0]
        expected = DataFrame([data_exp, foot_exp], columns=head_exp)
        expected = expected.fillna(np.nan)
        tm.assert_frame_equal(result, expected)

    def test_extract_links_bad(self, spam_data):
        msg = '`extract_links` must be one of {None, "header", "footer", "body", "all"}, got "incorrect"'
        with pytest.raises(ValueError, match=msg):
            read_html(spam_data, extract_links='incorrect')

    def test_extract_links_all_no_header(self, flavor_read_html):
        data = "\n        <table>\n          <tr>\n            <td>\n              <a href='https://google.com'>Google.com</a>\n            </td>\n          </tr>\n        </table>\n        "
        result = flavor_read_html(StringIO(data), extract_links='all')[0]
        expected = DataFrame([[('Google.com', 'https://google.com')]])
        tm.assert_frame_equal(result, expected)

    def test_invalid_dtype_backend(self):
        msg = "dtype_backend numpy is invalid, only 'numpy_nullable' and 'pyarrow' are allowed."
        with pytest.raises(ValueError, match=msg):
            read_html('test', dtype_backend='numpy')

    def test_style_tag(self, flavor_read_html):
        data = '\n        <table>\n            <tr>\n                <th>\n                    <style>.style</style>\n                    A\n                    </th>\n                <th>B</th>\n            </tr>\n            <tr>\n                <td>A1</td>\n                <td>B1</td>\n            </tr>\n            <tr>\n                <td>A2</td>\n                <td>B2</td>\n            </tr>\n        </table>\n        '
        result = flavor_read_html(StringIO(data))[0]
        expected = DataFrame(data=[['A1', 'B1'], ['A2', 'B2']], columns=['A', 'B'])
        tm.assert_frame_equal(result, expected)