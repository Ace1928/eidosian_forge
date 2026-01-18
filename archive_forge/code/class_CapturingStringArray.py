import numpy as np
import pytest
from pandas.core.dtypes import dtypes
from pandas.core.dtypes.common import is_extension_array_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
class CapturingStringArray(pd.arrays.StringArray):
    """Extend StringArray to capture arguments to __getitem__"""

    def __getitem__(self, item):
        self.last_item_arg = item
        return super().__getitem__(item)