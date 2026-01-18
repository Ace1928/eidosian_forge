from __future__ import annotations
import contextlib
import logging
import math
import shutil
import tempfile
import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal
import numpy as np
import pandas as pd
import tlz as toolz
from pandas.api.types import is_numeric_dtype
from dask import config
from dask.base import compute, compute_as_if_collection, is_dask_collection, tokenize
from dask.dataframe import methods
from dask.dataframe._compat import PANDAS_GE_300
from dask.dataframe.core import (
from dask.dataframe.dispatch import (
from dask.dataframe.utils import UNKNOWN_CATEGORIES
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ShuffleLayer, SimpleShuffleLayer
from dask.sizeof import sizeof
from dask.utils import M, digit, get_default_shuffle_method
def _calculate_divisions(df: DataFrame, partition_col: Series, repartition: bool, npartitions: int, upsample: float=1.0, partition_size: float=128000000.0, ascending: bool=True) -> tuple[list, list, list, bool]:
    """
    Utility function to calculate divisions for calls to `map_partitions`
    """
    sizes = df.map_partitions(sizeof) if repartition else []
    divisions = partition_col._repartition_quantiles(npartitions, upsample=upsample)
    mins = partition_col.map_partitions(M.min)
    maxes = partition_col.map_partitions(M.max)
    try:
        divisions, sizes, mins, maxes = compute(divisions, sizes, mins, maxes)
    except TypeError as e:
        if not is_numeric_dtype(partition_col.dtype):
            obj, suggested_method = ('column', f"`.dropna(subset=['{partition_col.name}'])`") if any((partition_col._name == df[c]._name for c in df)) else ('series', '`.loc[series[~series.isna()]]`')
            raise NotImplementedError(f"Divisions calculation failed for non-numeric {obj} '{partition_col.name}'.\nThis is probably due to the presence of nulls, which Dask does not entirely support in the index.\nWe suggest you try with {suggested_method}.") from e
        else:
            raise e
    empty_dataframe_detected = pd.isna(divisions).all()
    if repartition or empty_dataframe_detected:
        total = sum(sizes)
        npartitions = max(math.ceil(total / partition_size), 1)
        npartitions = min(npartitions, df.npartitions)
        n = divisions.size
        try:
            divisions = np.interp(x=np.linspace(0, n - 1, npartitions + 1), xp=np.linspace(0, n - 1, n), fp=divisions.tolist()).tolist()
        except (TypeError, ValueError):
            indexes = np.linspace(0, n - 1, npartitions + 1).astype(int)
            divisions = divisions.iloc[indexes].tolist()
    else:
        n = divisions.size
        divisions = list(divisions.iloc[:n - 1].unique()) + divisions.iloc[n - 1:].tolist()
    mins = mins.bfill()
    maxes = maxes.bfill()
    if isinstance(partition_col.dtype, pd.CategoricalDtype):
        dtype = partition_col.dtype
        mins = mins.astype(dtype)
        maxes = maxes.astype(dtype)
    if mins.isna().any() or maxes.isna().any():
        presorted = False
    else:
        n = mins.size
        maxes2 = (maxes.iloc[:n - 1] if ascending else maxes.iloc[1:]).reset_index(drop=True)
        mins2 = (mins.iloc[1:] if ascending else mins.iloc[:n - 1]).reset_index(drop=True)
        presorted = mins.tolist() == mins.sort_values(ascending=ascending).tolist() and maxes.tolist() == maxes.sort_values(ascending=ascending).tolist() and (maxes2 < mins2).all()
    return (divisions, mins.tolist(), maxes.tolist(), presorted)