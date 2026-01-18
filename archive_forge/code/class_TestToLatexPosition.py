import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestToLatexPosition:

    def test_to_latex_position(self):
        the_position = 'h'
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex(position=the_position)
        expected = _dedent('\n            \\begin{table}[h]\n            \\begin{tabular}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            \\end{table}\n            ')
        assert result == expected

    def test_to_latex_longtable_position(self):
        the_position = 't'
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex(longtable=True, position=the_position)
        expected = _dedent('\n            \\begin{longtable}[t]{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            \\endfirsthead\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            \\endhead\n            \\midrule\n            \\multicolumn{3}{r}{Continued on next page} \\\\\n            \\midrule\n            \\endfoot\n            \\bottomrule\n            \\endlastfoot\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\end{longtable}\n            ')
        assert result == expected