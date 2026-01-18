import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestToLatexLongtable:

    def test_to_latex_empty_longtable(self):
        df = DataFrame()
        result = df.to_latex(longtable=True)
        expected = _dedent('\n            \\begin{longtable}{l}\n            \\toprule\n            \\midrule\n            \\endfirsthead\n            \\toprule\n            \\midrule\n            \\endhead\n            \\midrule\n            \\multicolumn{0}{r}{Continued on next page} \\\\\n            \\midrule\n            \\endfoot\n            \\bottomrule\n            \\endlastfoot\n            \\end{longtable}\n            ')
        assert result == expected

    def test_to_latex_longtable_with_index(self):
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex(longtable=True)
        expected = _dedent('\n            \\begin{longtable}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            \\endfirsthead\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            \\endhead\n            \\midrule\n            \\multicolumn{3}{r}{Continued on next page} \\\\\n            \\midrule\n            \\endfoot\n            \\bottomrule\n            \\endlastfoot\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\end{longtable}\n            ')
        assert result == expected

    def test_to_latex_longtable_without_index(self):
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex(index=False, longtable=True)
        expected = _dedent('\n            \\begin{longtable}{rl}\n            \\toprule\n            a & b \\\\\n            \\midrule\n            \\endfirsthead\n            \\toprule\n            a & b \\\\\n            \\midrule\n            \\endhead\n            \\midrule\n            \\multicolumn{2}{r}{Continued on next page} \\\\\n            \\midrule\n            \\endfoot\n            \\bottomrule\n            \\endlastfoot\n            1 & b1 \\\\\n            2 & b2 \\\\\n            \\end{longtable}\n            ')
        assert result == expected

    @pytest.mark.parametrize('df, expected_number', [(DataFrame({'a': [1, 2]}), 1), (DataFrame({'a': [1, 2], 'b': [3, 4]}), 2), (DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}), 3)])
    def test_to_latex_longtable_continued_on_next_page(self, df, expected_number):
        result = df.to_latex(index=False, longtable=True)
        assert f'\\multicolumn{{{expected_number}}}' in result