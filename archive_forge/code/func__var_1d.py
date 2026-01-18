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
def _var_1d(self, column, skipna=True, ddof=1, split_every=False):
    is_timedelta = is_timedelta64_dtype(column._meta)
    if is_timedelta:
        if not skipna:
            is_nan = column.isna()
            column = column.astype('i8')
            column = column.mask(is_nan)
        else:
            column = column.dropna().astype('i8')
    if pd.api.types.is_extension_array_dtype(column._meta_nonempty):
        column = column.astype('f8')
    elif not np.issubdtype(column.dtype, np.number):
        column = column.astype('f8')
    name = self._token_prefix + 'var-1d-' + tokenize(column, split_every)
    var = da.nanvar if skipna or skipna is None else da.var
    array_var = var(column.values, axis=0, ddof=ddof, split_every=split_every)
    layer = {(name, 0): (methods.wrap_var_reduction, (array_var._name,), None)}
    graph = HighLevelGraph.from_collections(name, layer, dependencies=[array_var])
    return new_dd_object(graph, name, column._meta_nonempty.var(), divisions=[None, None])