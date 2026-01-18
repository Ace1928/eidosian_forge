from __future__ import annotations
import warnings
from functools import partial
import numpy as np
import pandas as pd
from pandas.api.types import is_extension_array_dtype
from pandas.errors import PerformanceWarning
from tlz import partition
from dask.dataframe._compat import (
from dask.dataframe.dispatch import (  # noqa: F401
from dask.dataframe.utils import is_dataframe_like, is_index_like, is_series_like
from dask.utils import _deprecated_kwarg
def _monotonic_chunk(x, prop):
    if x.empty:
        data = None
    else:
        data = x if is_index_like(x) else x.iloc
        data = [[getattr(x, prop), data[0], data[-1]]]
    return pd.DataFrame(data=data, columns=['monotonic', 'first', 'last'])