import operator
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def _check_unary_op(op):
    result = op(df1)
    expected = DataFrame(op(df1.values), index=df1.index, columns=df1.columns)
    assert result.values.dtype == np.bool_
    tm.assert_frame_equal(result, expected)