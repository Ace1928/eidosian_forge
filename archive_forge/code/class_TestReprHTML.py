from datetime import datetime
from io import StringIO
import itertools
import re
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
class TestReprHTML:

    def test_html_repr_min_rows_default(self, datapath):
        df = DataFrame({'a': range(20)})
        result = df._repr_html_()
        expected = expected_html(datapath, 'html_repr_min_rows_default_no_truncation')
        assert result == expected
        df = DataFrame({'a': range(61)})
        result = df._repr_html_()
        expected = expected_html(datapath, 'html_repr_min_rows_default_truncated')
        assert result == expected

    @pytest.mark.parametrize('max_rows,min_rows,expected', [(10, 4, 'html_repr_max_rows_10_min_rows_4'), (12, None, 'html_repr_max_rows_12_min_rows_None'), (10, 12, 'html_repr_max_rows_10_min_rows_12'), (None, 12, 'html_repr_max_rows_None_min_rows_12')])
    def test_html_repr_min_rows(self, datapath, max_rows, min_rows, expected):
        df = DataFrame({'a': range(61)})
        expected = expected_html(datapath, expected)
        with option_context('display.max_rows', max_rows, 'display.min_rows', min_rows):
            result = df._repr_html_()
        assert result == expected

    def test_repr_html_ipython_config(self, ip):
        code = textwrap.dedent('        from pandas import DataFrame\n        df = DataFrame({"A": [1, 2]})\n        df._repr_html_()\n\n        cfg = get_ipython().config\n        cfg[\'IPKernelApp\'][\'parent_appname\']\n        df._repr_html_()\n        ')
        result = ip.run_cell(code, silent=True)
        assert not result.error_in_exec

    def test_info_repr_html(self):
        max_rows = 60
        max_cols = 20
        h, w = (max_rows + 1, max_cols - 1)
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        assert '&lt;class' not in df._repr_html_()
        with option_context('display.large_repr', 'info'):
            assert '&lt;class' in df._repr_html_()
        h, w = (max_rows - 1, max_cols + 1)
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        assert '<class' not in df._repr_html_()
        with option_context('display.large_repr', 'info', 'display.max_columns', max_cols):
            assert '&lt;class' in df._repr_html_()

    def test_fake_qtconsole_repr_html(self, float_frame):
        df = float_frame

        def get_ipython():
            return {'config': {'KernelApp': {'parent_appname': 'ipython-qtconsole'}}}
        repstr = df._repr_html_()
        assert repstr is not None
        with option_context('display.max_rows', 5, 'display.max_columns', 2):
            repstr = df._repr_html_()
        assert 'class' in repstr

    def test_repr_html(self, float_frame):
        df = float_frame
        df._repr_html_()
        with option_context('display.max_rows', 1, 'display.max_columns', 1):
            df._repr_html_()
        with option_context('display.notebook_repr_html', False):
            df._repr_html_()
        df = DataFrame([[1, 2], [3, 4]])
        with option_context('display.show_dimensions', True):
            assert '2 rows' in df._repr_html_()
        with option_context('display.show_dimensions', False):
            assert '2 rows' not in df._repr_html_()

    def test_repr_html_mathjax(self):
        df = DataFrame([[1, 2], [3, 4]])
        assert 'tex2jax_ignore' not in df._repr_html_()
        with option_context('display.html.use_mathjax', False):
            assert 'tex2jax_ignore' in df._repr_html_()

    def test_repr_html_wide(self):
        max_cols = 20
        df = DataFrame([['a' * 25] * (max_cols - 1)] * 10)
        with option_context('display.max_rows', 60, 'display.max_columns', 20):
            assert '...' not in df._repr_html_()
        wide_df = DataFrame([['a' * 25] * (max_cols + 1)] * 10)
        with option_context('display.max_rows', 60, 'display.max_columns', 20):
            assert '...' in wide_df._repr_html_()

    def test_repr_html_wide_multiindex_cols(self):
        max_cols = 20
        mcols = MultiIndex.from_product([np.arange(max_cols // 2), ['foo', 'bar']], names=['first', 'second'])
        df = DataFrame([['a' * 25] * len(mcols)] * 10, columns=mcols)
        reg_repr = df._repr_html_()
        assert '...' not in reg_repr
        mcols = MultiIndex.from_product((np.arange(1 + max_cols // 2), ['foo', 'bar']), names=['first', 'second'])
        df = DataFrame([['a' * 25] * len(mcols)] * 10, columns=mcols)
        with option_context('display.max_rows', 60, 'display.max_columns', 20):
            assert '...' in df._repr_html_()

    def test_repr_html_long(self):
        with option_context('display.max_rows', 60):
            max_rows = get_option('display.max_rows')
            h = max_rows - 1
            df = DataFrame({'A': np.arange(1, 1 + h), 'B': np.arange(41, 41 + h)})
            reg_repr = df._repr_html_()
            assert '..' not in reg_repr
            assert str(41 + max_rows // 2) in reg_repr
            h = max_rows + 1
            df = DataFrame({'A': np.arange(1, 1 + h), 'B': np.arange(41, 41 + h)})
            long_repr = df._repr_html_()
            assert '..' in long_repr
            assert str(41 + max_rows // 2) not in long_repr
            assert f'{h} rows ' in long_repr
            assert '2 columns' in long_repr

    def test_repr_html_float(self):
        with option_context('display.max_rows', 60):
            max_rows = get_option('display.max_rows')
            h = max_rows - 1
            df = DataFrame({'idx': np.linspace(-10, 10, h), 'A': np.arange(1, 1 + h), 'B': np.arange(41, 41 + h)}).set_index('idx')
            reg_repr = df._repr_html_()
            assert '..' not in reg_repr
            assert f'<td>{40 + h}</td>' in reg_repr
            h = max_rows + 1
            df = DataFrame({'idx': np.linspace(-10, 10, h), 'A': np.arange(1, 1 + h), 'B': np.arange(41, 41 + h)}).set_index('idx')
            long_repr = df._repr_html_()
            assert '..' in long_repr
            assert '<td>31</td>' not in long_repr
            assert f'{h} rows ' in long_repr
            assert '2 columns' in long_repr

    def test_repr_html_long_multiindex(self):
        max_rows = 60
        max_L1 = max_rows // 2
        tuples = list(itertools.product(np.arange(max_L1), ['foo', 'bar']))
        idx = MultiIndex.from_tuples(tuples, names=['first', 'second'])
        df = DataFrame(np.random.default_rng(2).standard_normal((max_L1 * 2, 2)), index=idx, columns=['A', 'B'])
        with option_context('display.max_rows', 60, 'display.max_columns', 20):
            reg_repr = df._repr_html_()
        assert '...' not in reg_repr
        tuples = list(itertools.product(np.arange(max_L1 + 1), ['foo', 'bar']))
        idx = MultiIndex.from_tuples(tuples, names=['first', 'second'])
        df = DataFrame(np.random.default_rng(2).standard_normal(((max_L1 + 1) * 2, 2)), index=idx, columns=['A', 'B'])
        long_repr = df._repr_html_()
        assert '...' in long_repr

    def test_repr_html_long_and_wide(self):
        max_cols = 20
        max_rows = 60
        h, w = (max_rows - 1, max_cols - 1)
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        with option_context('display.max_rows', 60, 'display.max_columns', 20):
            assert '...' not in df._repr_html_()
        h, w = (max_rows + 1, max_cols + 1)
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        with option_context('display.max_rows', 60, 'display.max_columns', 20):
            assert '...' in df._repr_html_()