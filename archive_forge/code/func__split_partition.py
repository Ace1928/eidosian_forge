from __future__ import annotations
import math
import pickle
import warnings
from functools import partial, wraps
import numpy as np
import pandas as pd
from pandas.api.types import is_dtype_equal
from tlz import merge_sorted, unique
from dask.base import is_dask_collection, tokenize
from dask.dataframe import methods
from dask.dataframe.core import (
from dask.dataframe.dispatch import group_split_dispatch, hash_object_dispatch
from dask.dataframe.io import from_pandas
from dask.dataframe.shuffle import (
from dask.dataframe.utils import (
from dask.highlevelgraph import HighLevelGraph
from dask.layers import BroadcastJoinLayer
from dask.utils import M, apply, get_default_shuffle_method
def _split_partition(df, on, nsplits):
    """
    Split-by-hash a DataFrame into `nsplits` groups.

    Hashing will be performed on the columns or index specified by `on`.
    """
    if isinstance(on, bytes):
        on = pickle.loads(on)
    if isinstance(on, str) or pd.api.types.is_list_like(on):
        on = [on] if isinstance(on, str) else list(on)
        nset = set(on)
        if nset.intersection(set(df.columns)) == nset:
            o = df[on]
            dtypes = {}
            for col, dtype in o.dtypes.items():
                if pd.api.types.is_numeric_dtype(dtype):
                    dtypes[col] = np.float64
            if not dtypes:
                ind = hash_object_dispatch(df[on], index=False)
            else:
                ind = hash_object_dispatch(df[on].astype(dtypes), index=False)
            ind = ind % nsplits
            return group_split_dispatch(df, ind, nsplits, ignore_index=False)
    if not isinstance(on, _Frame):
        on = _select_columns_or_index(df, on)
    dtypes = {}
    for col, dtype in on.dtypes.items():
        if pd.api.types.is_numeric_dtype(dtype):
            dtypes[col] = np.float64
    if not dtypes:
        dtypes = None
    partitions = partitioning_index(on, nsplits, cast_dtype=dtypes)
    df2 = df.assign(_partitions=partitions)
    return shuffle_group(df2, ['_partitions'], 0, nsplits, nsplits, False, nsplits)