import collections
from functools import partial
import string
import subprocess
import sys
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
from pandas.core import ops
import pandas.core.common as com
from pandas.util.version import Version
class TestIsBoolIndexer:

    def test_non_bool_array_with_na(self):
        arr = np.array(['A', 'B', np.nan], dtype=object)
        assert not com.is_bool_indexer(arr)

    def test_list_subclass(self):

        class MyList(list):
            pass
        val = MyList(['a'])
        assert not com.is_bool_indexer(val)
        val = MyList([True])
        assert com.is_bool_indexer(val)

    def test_frozenlist(self):
        data = {'col1': [1, 2], 'col2': [3, 4]}
        df = pd.DataFrame(data=data)
        frozen = df.index.names[1:]
        assert not com.is_bool_indexer(frozen)
        result = df[frozen]
        expected = df[[]]
        tm.assert_frame_equal(result, expected)