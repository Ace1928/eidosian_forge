from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestDataFrameIndexingUInt64:

    def test_setitem(self):
        df = DataFrame({'A': np.arange(3), 'B': [2 ** 63, 2 ** 63 + 5, 2 ** 63 + 10]}, dtype=np.uint64)
        idx = df['A'].rename('foo')
        assert 'C' not in df.columns
        df['C'] = idx
        tm.assert_series_equal(df['C'], Series(idx, name='C'))
        assert 'D' not in df.columns
        df['D'] = 'foo'
        df['D'] = idx
        tm.assert_series_equal(df['D'], Series(idx, name='D'))
        del df['D']
        df2 = df.copy()
        with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
            df2.iloc[1, 1] = pd.NaT
            df2.iloc[1, 2] = pd.NaT
        result = df2['B']
        tm.assert_series_equal(notna(result), Series([True, False, True], name='B'))
        tm.assert_series_equal(df2.dtypes, Series([np.dtype('uint64'), np.dtype('O'), np.dtype('O')], index=['A', 'B', 'C']))