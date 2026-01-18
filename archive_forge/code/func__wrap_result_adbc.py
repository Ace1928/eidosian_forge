from __future__ import annotations
from abc import (
from contextlib import (
from datetime import (
from functools import partial
import re
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas import get_option
from pandas.core.api import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.common import maybe_make_list
from pandas.core.internals.construction import convert_object_array
from pandas.core.tools.datetimes import to_datetime
def _wrap_result_adbc(df: DataFrame, *, index_col=None, parse_dates=None, dtype: DtypeArg | None=None, dtype_backend: DtypeBackend | Literal['numpy']='numpy') -> DataFrame:
    """Wrap result set of a SQLAlchemy query in a DataFrame."""
    if dtype:
        df = df.astype(dtype)
    df = _parse_date_columns(df, parse_dates)
    if index_col is not None:
        df = df.set_index(index_col)
    return df