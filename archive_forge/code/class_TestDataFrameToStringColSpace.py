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
class TestDataFrameToStringColSpace:

    def test_to_string_with_column_specific_col_space_raises(self):
        df = DataFrame(np.random.default_rng(2).random(size=(3, 3)), columns=['a', 'b', 'c'])
        msg = 'Col_space length\\(\\d+\\) should match DataFrame number of columns\\(\\d+\\)'
        with pytest.raises(ValueError, match=msg):
            df.to_string(col_space=[30, 40])
        with pytest.raises(ValueError, match=msg):
            df.to_string(col_space=[30, 40, 50, 60])
        msg = 'unknown column'
        with pytest.raises(ValueError, match=msg):
            df.to_string(col_space={'a': 'foo', 'b': 23, 'd': 34})

    def test_to_string_with_column_specific_col_space(self):
        df = DataFrame(np.random.default_rng(2).random(size=(3, 3)), columns=['a', 'b', 'c'])
        result = df.to_string(col_space={'a': 10, 'b': 11, 'c': 12})
        assert len(result.split('\n')[1]) == 3 + 1 + 10 + 11 + 12
        result = df.to_string(col_space=[10, 11, 12])
        assert len(result.split('\n')[1]) == 3 + 1 + 10 + 11 + 12

    def test_to_string_with_col_space(self):
        df = DataFrame(np.random.default_rng(2).random(size=(1, 3)))
        c10 = len(df.to_string(col_space=10).split('\n')[1])
        c20 = len(df.to_string(col_space=20).split('\n')[1])
        c30 = len(df.to_string(col_space=30).split('\n')[1])
        assert c10 < c20 < c30
        with_header = df.to_string(col_space=20)
        with_header_row1 = with_header.splitlines()[1]
        no_header = df.to_string(col_space=20, header=False)
        assert len(with_header_row1) == len(no_header)

    def test_to_string_repr_tuples(self):
        buf = StringIO()
        df = DataFrame({'tups': list(zip(range(10), range(10)))})
        repr(df)
        df.to_string(col_space=10, buf=buf)