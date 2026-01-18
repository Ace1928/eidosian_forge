import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def _check_replace_with_method(self, ser: pd.Series):
    df = ser.to_frame()
    msg1 = "The 'method' keyword in Series.replace is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg1):
        res = ser.replace(ser[1], method='pad')
    expected = pd.Series([ser[0], ser[0]] + list(ser[2:]), dtype=ser.dtype)
    tm.assert_series_equal(res, expected)
    msg2 = "The 'method' keyword in DataFrame.replace is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg2):
        res_df = df.replace(ser[1], method='pad')
    tm.assert_frame_equal(res_df, expected.to_frame())
    ser2 = ser.copy()
    with tm.assert_produces_warning(FutureWarning, match=msg1):
        res2 = ser2.replace(ser[1], method='pad', inplace=True)
    assert res2 is None
    tm.assert_series_equal(ser2, expected)
    with tm.assert_produces_warning(FutureWarning, match=msg2):
        res_df2 = df.replace(ser[1], method='pad', inplace=True)
    assert res_df2 is None
    tm.assert_frame_equal(df, expected.to_frame())