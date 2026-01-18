from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def _check_align(df, cond, other, check_dtypes=True):
    rs = df.where(cond, other)
    for i, k in enumerate(rs.columns):
        result = rs[k]
        d = df[k].values
        c = cond[k].reindex(df[k].index).fillna(False).values
        if is_scalar(other):
            o = other
        elif isinstance(other, np.ndarray):
            o = Series(other[:, i], index=result.index).values
        else:
            o = other[k].values
        new_values = d if c.all() else np.where(c, d, o)
        expected = Series(new_values, index=result.index, name=k)
        tm.assert_series_equal(result, expected, check_dtype=False)
    if check_dtypes and (not isinstance(other, np.ndarray)):
        assert (rs.dtypes == df.dtypes).all()