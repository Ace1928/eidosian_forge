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
class TestDataFrameToStringHeader:

    def test_to_string_header_false(self):
        df = DataFrame([1, 2])
        df.index.name = 'a'
        s = df.to_string(header=False)
        expected = 'a   \n0  1\n1  2'
        assert s == expected
        df = DataFrame([[1, 2], [3, 4]])
        df.index.name = 'a'
        s = df.to_string(header=False)
        expected = 'a      \n0  1  2\n1  3  4'
        assert s == expected

    def test_to_string_multindex_header(self):
        df = DataFrame({'a': [0], 'b': [1], 'c': [2], 'd': [3]}).set_index(['a', 'b'])
        res = df.to_string(header=['r1', 'r2'])
        exp = '    r1 r2\na b      \n0 1  2  3'
        assert res == exp

    def test_to_string_no_header(self):
        df = DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        df_s = df.to_string(header=False)
        expected = '0  1  4\n1  2  5\n2  3  6'
        assert df_s == expected

    def test_to_string_specified_header(self):
        df = DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        df_s = df.to_string(header=['X', 'Y'])
        expected = '   X  Y\n0  1  4\n1  2  5\n2  3  6'
        assert df_s == expected
        msg = 'Writing 2 cols but got 1 aliases'
        with pytest.raises(ValueError, match=msg):
            df.to_string(header=['X'])