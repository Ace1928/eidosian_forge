from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def _check_get(df, cond, check_dtypes=True):
    other1 = _safe_add(df)
    rs = df.where(cond, other1)
    rs2 = df.where(cond.values, other1)
    for k, v in rs.items():
        exp = Series(np.where(cond[k], df[k], other1[k]), index=v.index)
        tm.assert_series_equal(v, exp, check_names=False)
    tm.assert_frame_equal(rs, rs2)
    if check_dtypes:
        assert (rs.dtypes == df.dtypes).all()