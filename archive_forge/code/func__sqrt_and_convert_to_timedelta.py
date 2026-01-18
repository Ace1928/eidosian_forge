from __future__ import annotations
import operator
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial, wraps
from numbers import Integral, Number
from operator import getitem
from pprint import pformat
from typing import Any, ClassVar, Literal, cast
import numpy as np
import pandas as pd
from pandas.api.types import (
from tlz import first, merge, partition_all, remove, unique
import dask.array as da
from dask import config, core
from dask.array.core import Array, normalize_arg
from dask.bag import map_partitions as map_bag_partitions
from dask.base import (
from dask.blockwise import Blockwise, BlockwiseDep, BlockwiseDepDict, blockwise
from dask.context import globalmethod
from dask.dataframe import methods
from dask.dataframe._compat import (
from dask.dataframe.accessor import CachedAccessor, DatetimeAccessor, StringAccessor
from dask.dataframe.categorical import CategoricalAccessor, categorize
from dask.dataframe.dispatch import (
from dask.dataframe.optimize import optimize
from dask.dataframe.utils import (
from dask.delayed import Delayed, delayed, unpack_collections
from dask.highlevelgraph import HighLevelGraph
from dask.layers import DataFrameTreeReduction
from dask.typing import Graph, NestedKeys, no_default
from dask.utils import (
from dask.widgets import get_template
def _sqrt_and_convert_to_timedelta(partition, axis, dtype=None, *args, **kwargs):
    if axis == 1:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in cast')
            return pd.to_timedelta(M.std(partition, *args, axis=axis, **kwargs))
    is_df_like, time_cols = (kwargs['is_df_like'], kwargs['time_cols'])
    sqrt = np.sqrt(partition)
    if not is_df_like:
        return pd.to_timedelta(sqrt)
    time_col_mask = sqrt.index.isin(time_cols)
    matching_vals = sqrt[time_col_mask]
    if len(time_cols) > 0:
        sqrt = sqrt.astype(object)
    for time_col, matching_val in zip(time_cols, matching_vals):
        sqrt[time_col] = pd.to_timedelta(matching_val)
    if dtype is not None:
        sqrt = sqrt.astype(dtype)
    return sqrt