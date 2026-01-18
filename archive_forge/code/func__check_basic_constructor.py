import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
def _check_basic_constructor(self, empty):
    mat = empty((2, 3), dtype=float)
    frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
    assert len(frame.index) == 2
    assert len(frame.columns) == 3
    frame = DataFrame(empty((3,)), columns=['A'], index=[1, 2, 3])
    assert len(frame.index) == 3
    assert len(frame.columns) == 1
    if empty is not np.ones:
        msg = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
        with pytest.raises(IntCastingNaNError, match=msg):
            DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2], dtype=np.int64)
        return
    else:
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2], dtype=np.int64)
        assert frame.values.dtype == np.int64
    msg = 'Shape of passed values is \\(2, 3\\), indices imply \\(1, 3\\)'
    with pytest.raises(ValueError, match=msg):
        DataFrame(mat, columns=['A', 'B', 'C'], index=[1])
    msg = 'Shape of passed values is \\(2, 3\\), indices imply \\(2, 2\\)'
    with pytest.raises(ValueError, match=msg):
        DataFrame(mat, columns=['A', 'B'], index=[1, 2])
    with pytest.raises(ValueError, match='Must pass 2-d input'):
        DataFrame(empty((3, 3, 3)), columns=['A', 'B', 'C'], index=[1])
    frame = DataFrame(mat)
    tm.assert_index_equal(frame.index, Index(range(2)), exact=True)
    tm.assert_index_equal(frame.columns, Index(range(3)), exact=True)
    frame = DataFrame(mat, index=[1, 2])
    tm.assert_index_equal(frame.columns, Index(range(3)), exact=True)
    frame = DataFrame(mat, columns=['A', 'B', 'C'])
    tm.assert_index_equal(frame.index, Index(range(2)), exact=True)
    frame = DataFrame(empty((0, 3)))
    assert len(frame.index) == 0
    frame = DataFrame(empty((3, 0)))
    assert len(frame.columns) == 0