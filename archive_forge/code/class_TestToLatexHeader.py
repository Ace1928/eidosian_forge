import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestToLatexHeader:

    def test_to_latex_no_header_with_index(self):
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex(header=False)
        expected = _dedent('\n            \\begin{tabular}{lrl}\n            \\toprule\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_no_header_without_index(self):
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex(index=False, header=False)
        expected = _dedent('\n            \\begin{tabular}{rl}\n            \\toprule\n            \\midrule\n            1 & b1 \\\\\n            2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_specified_header_with_index(self):
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex(header=['AA', 'BB'])
        expected = _dedent('\n            \\begin{tabular}{lrl}\n            \\toprule\n             & AA & BB \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_specified_header_without_index(self):
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex(header=['AA', 'BB'], index=False)
        expected = _dedent('\n            \\begin{tabular}{rl}\n            \\toprule\n            AA & BB \\\\\n            \\midrule\n            1 & b1 \\\\\n            2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    @pytest.mark.parametrize('header, num_aliases', [(['A'], 1), (('B',), 1), (('Col1', 'Col2', 'Col3'), 3), (('Col1', 'Col2', 'Col3', 'Col4'), 4)])
    def test_to_latex_number_of_items_in_header_missmatch_raises(self, header, num_aliases):
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        msg = f'Writing 2 cols but got {num_aliases} aliases'
        with pytest.raises(ValueError, match=msg):
            df.to_latex(header=header)

    def test_to_latex_decimal(self):
        df = DataFrame({'a': [1.0, 2.1], 'b': ['b1', 'b2']})
        result = df.to_latex(decimal=',')
        expected = _dedent('\n            \\begin{tabular}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            0 & 1,000000 & b1 \\\\\n            1 & 2,100000 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected