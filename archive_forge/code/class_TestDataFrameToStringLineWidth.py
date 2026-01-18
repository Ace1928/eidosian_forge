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
class TestDataFrameToStringLineWidth:

    def test_to_string_line_width(self):
        df = DataFrame(123, index=range(10, 15), columns=range(30))
        lines = df.to_string(line_width=80)
        assert max((len(line) for line in lines.split('\n'))) == 80

    def test_to_string_line_width_no_index(self):
        df = DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        df_s = df.to_string(line_width=1, index=False)
        expected = ' x  \\\n 1   \n 2   \n 3   \n\n y  \n 4  \n 5  \n 6  '
        assert df_s == expected
        df = DataFrame({'x': [11, 22, 33], 'y': [4, 5, 6]})
        df_s = df.to_string(line_width=1, index=False)
        expected = ' x  \\\n11   \n22   \n33   \n\n y  \n 4  \n 5  \n 6  '
        assert df_s == expected
        df = DataFrame({'x': [11, 22, -33], 'y': [4, 5, -6]})
        df_s = df.to_string(line_width=1, index=False)
        expected = '  x  \\\n 11   \n 22   \n-33   \n\n y  \n 4  \n 5  \n-6  '
        assert df_s == expected

    def test_to_string_line_width_no_header(self):
        df = DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        df_s = df.to_string(line_width=1, header=False)
        expected = '0  1  \\\n1  2   \n2  3   \n\n0  4  \n1  5  \n2  6  '
        assert df_s == expected
        df = DataFrame({'x': [11, 22, 33], 'y': [4, 5, 6]})
        df_s = df.to_string(line_width=1, header=False)
        expected = '0  11  \\\n1  22   \n2  33   \n\n0  4  \n1  5  \n2  6  '
        assert df_s == expected
        df = DataFrame({'x': [11, 22, -33], 'y': [4, 5, -6]})
        df_s = df.to_string(line_width=1, header=False)
        expected = '0  11  \\\n1  22   \n2 -33   \n\n0  4  \n1  5  \n2 -6  '
        assert df_s == expected

    def test_to_string_line_width_with_both_index_and_header(self):
        df = DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        df_s = df.to_string(line_width=1)
        expected = '   x  \\\n0  1   \n1  2   \n2  3   \n\n   y  \n0  4  \n1  5  \n2  6  '
        assert df_s == expected
        df = DataFrame({'x': [11, 22, 33], 'y': [4, 5, 6]})
        df_s = df.to_string(line_width=1)
        expected = '    x  \\\n0  11   \n1  22   \n2  33   \n\n   y  \n0  4  \n1  5  \n2  6  '
        assert df_s == expected
        df = DataFrame({'x': [11, 22, -33], 'y': [4, 5, -6]})
        df_s = df.to_string(line_width=1)
        expected = '    x  \\\n0  11   \n1  22   \n2 -33   \n\n   y  \n0  4  \n1  5  \n2 -6  '
        assert df_s == expected

    def test_to_string_line_width_no_index_no_header(self):
        df = DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        df_s = df.to_string(line_width=1, index=False, header=False)
        expected = '1  \\\n2   \n3   \n\n4  \n5  \n6  '
        assert df_s == expected
        df = DataFrame({'x': [11, 22, 33], 'y': [4, 5, 6]})
        df_s = df.to_string(line_width=1, index=False, header=False)
        expected = '11  \\\n22   \n33   \n\n4  \n5  \n6  '
        assert df_s == expected
        df = DataFrame({'x': [11, 22, -33], 'y': [4, 5, -6]})
        df_s = df.to_string(line_width=1, index=False, header=False)
        expected = ' 11  \\\n 22   \n-33   \n\n 4  \n 5  \n-6  '
        assert df_s == expected