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
def _convert_time_cols_to_numeric(self, time_cols, axis, meta, skipna):
    from dask.dataframe.io import from_pandas
    needs_time_conversion = True
    if axis == 0:
        numeric_dd = self[meta.index].copy()
    else:
        numeric_dd = self.copy()
    if axis == 1 and len(time_cols) != len(self.columns):
        needs_time_conversion = False
        numeric_dd = from_pandas(meta_frame_constructor(self)({'_': meta_series_constructor(self)([np.nan])}, index=self.index), npartitions=self.npartitions)
    else:
        for col in time_cols:
            numeric_dd[col] = _convert_to_numeric(numeric_dd[col], skipna)
    return (numeric_dd, needs_time_conversion)