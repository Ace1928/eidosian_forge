import numpy as np
import pytest
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import BaseMaskedArray
class MyStringArray(BaseMaskedArray):
    dtype = pd.StringDtype()
    _dtype_cls = pd.StringDtype
    _internal_fill_value = pd.NA