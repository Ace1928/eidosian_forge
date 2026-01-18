from __future__ import annotations
from functools import partial
import pandas as pd
from packaging.version import Version
from dask.dataframe._compat import PANDAS_GE_150, PANDAS_GE_200
from dask.dataframe.utils import is_dataframe_like, is_index_like, is_series_like
def _to_string_dtype(df, dtype_check, index_check, string_dtype):
    if not (is_dataframe_like(df) or is_series_like(df) or is_index_like(df)):
        return df
    if string_dtype == 'pyarrow':
        string_dtype = pd.StringDtype('pyarrow')
    if is_dataframe_like(df):
        dtypes = {col: string_dtype for col, dtype in df.dtypes.items() if dtype_check(dtype)}
        if dtypes:
            df = df.astype(dtypes)
    elif dtype_check(df.dtype):
        dtypes = string_dtype
        df = df.copy().astype(dtypes)
    if (is_dataframe_like(df) or is_series_like(df)) and index_check(df.index):
        if isinstance(df.index, pd.MultiIndex):
            levels = {i: level.astype(string_dtype) for i, level in enumerate(df.index.levels) if dtype_check(level.dtype)}
            df.index = df.index.set_levels(levels.values(), level=levels.keys(), verify_integrity=False)
        else:
            df.index = df.index.astype(string_dtype)
    return df