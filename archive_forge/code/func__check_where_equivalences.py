from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def _check_where_equivalences(df, mask, other, expected):
    res = df.where(mask, other)
    tm.assert_frame_equal(res, expected)
    res = df.mask(~mask, other)
    tm.assert_frame_equal(res, expected)
    df = df.copy()
    df.mask(~mask, other, inplace=True)
    if not mask.all():
        expected = expected.copy()
        expected['A'] = expected['A'].astype(object)
    tm.assert_frame_equal(df, expected)