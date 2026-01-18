from __future__ import annotations
import warnings
from collections.abc import Iterable
import numpy as np
import pandas as pd
from pandas.api.types import is_scalar, union_categoricals
from dask.array.core import Array
from dask.array.dispatch import percentile_lookup
from dask.array.percentile import _percentile
from dask.backends import CreationDispatch, DaskBackendEntrypoint
from dask.dataframe._compat import PANDAS_GE_220, is_any_real_numeric_dtype
from dask.dataframe.core import DataFrame, Index, Scalar, Series, _Frame
from dask.dataframe.dispatch import (
from dask.dataframe.extensions import make_array_nonempty, make_scalar
from dask.dataframe.utils import (
from dask.sizeof import SimpleSizeof, sizeof
from dask.utils import is_arraylike, is_series_like, typename
@meta_nonempty.register(pd.Index)
def _nonempty_index(idx):
    typ = type(idx)
    if typ is pd.RangeIndex:
        return pd.RangeIndex(2, name=idx.name, dtype=idx.dtype)
    elif is_any_real_numeric_dtype(idx):
        return typ([1, 2], name=idx.name, dtype=idx.dtype)
    elif typ is pd.DatetimeIndex:
        start = '1970-01-01'
        try:
            return pd.date_range(start=start, periods=2, freq=idx.freq, tz=idx.tz, name=idx.name)
        except ValueError:
            data = [start, '1970-01-02'] if idx.freq is None else None
            return pd.DatetimeIndex(data, start=start, periods=2, freq=idx.freq, tz=idx.tz, name=idx.name)
    elif typ is pd.PeriodIndex:
        return pd.period_range(start='1970-01-01', periods=2, freq=idx.freq, name=idx.name)
    elif typ is pd.TimedeltaIndex:
        start = np.timedelta64(1, 'D')
        try:
            return pd.timedelta_range(start=start, periods=2, freq=idx.freq, name=idx.name)
        except ValueError:
            start = np.timedelta64(1, 'D')
            data = [start, start + 1] if idx.freq is None else None
            return pd.TimedeltaIndex(data, start=start, periods=2, freq=idx.freq, name=idx.name)
    elif typ is pd.CategoricalIndex:
        if len(idx.categories) == 0:
            data = pd.Categorical(_nonempty_index(idx.categories), ordered=idx.ordered)
        else:
            data = pd.Categorical.from_codes([-1, 0], categories=idx.categories, ordered=idx.ordered)
        return pd.CategoricalIndex(data, name=idx.name)
    elif typ is pd.MultiIndex:
        levels = [_nonempty_index(l) for l in idx.levels]
        codes = [[0, 0] for i in idx.levels]
        try:
            return pd.MultiIndex(levels=levels, codes=codes, names=idx.names)
        except TypeError:
            return pd.MultiIndex(levels=levels, labels=codes, names=idx.names)
    elif typ is pd.Index:
        if type(idx.dtype) in make_array_nonempty._lookup:
            return pd.Index(make_array_nonempty(idx.dtype), dtype=idx.dtype, name=idx.name)
        elif idx.dtype == bool:
            return pd.Index([True, False], name=idx.name)
        else:
            return pd.Index(['a', 'b'], name=idx.name, dtype=idx.dtype)
    raise TypeError(f"Don't know how to handle index of type {typename(type(idx))}")