from __future__ import annotations
from datetime import (
import itertools
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas as pd
import pandas._testing as tm
def _construct_exp(self, obj, klass, fill_val, exp_dtype):
    if fill_val is True:
        values = klass([True, False, True, True])
    elif isinstance(fill_val, (datetime, np.datetime64)):
        values = pd.date_range(fill_val, periods=4)
    else:
        values = klass((x * fill_val for x in [5, 6, 7, 8]))
    exp = klass([obj[0], values[1], obj[2], values[3]], dtype=exp_dtype)
    return (values, exp)