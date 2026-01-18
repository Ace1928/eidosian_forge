from __future__ import annotations
from datetime import (
from typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
import pandas as pd
from pandas.core.shared_docs import _shared_docs
from pandas.io.excel._base import BaseExcelReader
def _convert_cell(value: _CellValue) -> Scalar | NaTType | time:
    if isinstance(value, float):
        val = int(value)
        if val == value:
            return val
        else:
            return value
    elif isinstance(value, date):
        return pd.Timestamp(value)
    elif isinstance(value, timedelta):
        return pd.Timedelta(value)
    elif isinstance(value, time):
        return value
    return value