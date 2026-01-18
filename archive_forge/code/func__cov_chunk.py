from __future__ import annotations
import collections
import itertools as it
import operator
import uuid
import warnings
from functools import partial, wraps
from numbers import Integral
import numpy as np
import pandas as pd
from dask.base import is_dask_collection, tokenize
from dask.core import flatten
from dask.dataframe._compat import (
from dask.dataframe.core import (
from dask.dataframe.dispatch import grouper_dispatch
from dask.dataframe.methods import concat, drop_columns
from dask.dataframe.utils import (
from dask.highlevelgraph import HighLevelGraph
from dask.typing import no_default
from dask.utils import (
def _cov_chunk(df, *by, numeric_only=no_default):
    """Covariance Chunk Logic

    Parameters
    ----------
    df : Pandas.DataFrame
    std : bool, optional
        When std=True we are calculating with Correlation

    Returns
    -------
    tuple
        Processed X, Multiplied Cols,
    """
    numeric_only_kwargs = get_numeric_only_kwargs(numeric_only)
    if is_series_like(df):
        df = df.to_frame()
    df = df.copy()
    if numeric_only is False:
        dt_df = df.select_dtypes(include=['datetime', 'timedelta'])
        for col in dt_df.columns:
            df[col] = _convert_to_numeric(dt_df[col], True)
    col_mapping = collections.OrderedDict()
    for i, c in enumerate(df.columns):
        col_mapping[c] = str(i)
    df = df.rename(columns=col_mapping)
    cols = df._get_numeric_data().columns
    is_mask = any((is_series_like(s) for s in by))
    if not is_mask:
        by = [col_mapping[k] for k in by]
        cols = cols.difference(pd.Index(by))
    g = _groupby_raise_unaligned(df, by=by)
    x = g.sum(**numeric_only_kwargs)
    include_groups = {'include_groups': False} if PANDAS_GE_220 else {}
    mul = g.apply(_mul_cols, cols=cols, **include_groups).reset_index(level=-1, drop=True)
    n = g[x.columns].count().rename(columns=lambda c: f'{c}-count')
    return (x, mul, n, col_mapping)