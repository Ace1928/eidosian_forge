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
def _compute_partition_stats(column: Series, allow_overlap: bool=False, **kwargs) -> tuple[list, list, list[int]]:
    """For a given column, compute the min, max, and len of each partition.

    And make sure that the partitions are sorted relative to each other.
    NOTE: this does not guarantee that every partition is internally sorted.
    """
    mins = column.map_partitions(M.min, meta=column)
    maxes = column.map_partitions(M.max, meta=column)
    lens = column.map_partitions(len, meta=column)
    mins, maxes, lens = compute(mins, maxes, lens, **kwargs)
    mins = mins.bfill().tolist()
    maxes = maxes.bfill().tolist()
    non_empty_mins = [m for m, length in zip(mins, lens) if length != 0]
    non_empty_maxes = [m for m, length in zip(maxes, lens) if length != 0]
    if sorted(non_empty_mins) != non_empty_mins or sorted(non_empty_maxes) != non_empty_maxes:
        raise ValueError(f'Partitions are not sorted ascending by {column.name or 'the index'}', f'In your dataset the (min, max, len) values of {column.name or 'the index'} for each partition are : {list(zip(mins, maxes, lens))}')
    if not allow_overlap and any((a <= b for a, b in zip(non_empty_mins[1:], non_empty_maxes[:-1]))):
        warnings.warn(f'Partitions have overlapping values, so divisions are non-unique.Use `set_index(sorted=True)` with no `divisions` to allow dask to fix the overlap. In your dataset the (min, max, len) values of {column.name or 'the index'} for each partition are : {list(zip(mins, maxes, lens))}', UserWarning)
    lens = methods.tolist(lens)
    if not allow_overlap:
        return (mins, maxes, lens)
    else:
        return (non_empty_mins, non_empty_maxes, lens)