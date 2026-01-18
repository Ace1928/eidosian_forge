import inspect
import operator
import numpy as np
import pytest
from pandas._typing import Dtype
from pandas.core.dtypes.common import is_bool_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.sorting import nargsort
def _test_searchsorted_bool_dtypes(self, data_for_sorting, as_series):
    dtype = data_for_sorting.dtype
    data_for_sorting = pd.array([True, False], dtype=dtype)
    b, a = data_for_sorting
    arr = type(data_for_sorting)._from_sequence([a, b])
    if as_series:
        arr = pd.Series(arr)
    assert arr.searchsorted(a) == 0
    assert arr.searchsorted(a, side='right') == 1
    assert arr.searchsorted(b) == 1
    assert arr.searchsorted(b, side='right') == 2
    result = arr.searchsorted(arr.take([0, 1]))
    expected = np.array([0, 1], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)
    sorter = np.array([1, 0])
    assert data_for_sorting.searchsorted(a, sorter=sorter) == 0