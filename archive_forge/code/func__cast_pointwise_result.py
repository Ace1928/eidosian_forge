from __future__ import annotations
from typing import final
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core import ops
def _cast_pointwise_result(self, op_name: str, obj, other, pointwise_result):
    return pointwise_result