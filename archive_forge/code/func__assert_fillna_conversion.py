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
def _assert_fillna_conversion(self, original, value, expected, expected_dtype):
    """test coercion triggered by fillna"""
    target = original.copy()
    res = target.fillna(value)
    tm.assert_equal(res, expected)
    assert res.dtype == expected_dtype