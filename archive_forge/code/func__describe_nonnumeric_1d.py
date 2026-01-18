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
def _describe_nonnumeric_1d(self, data, split_every=False, datetime_is_numeric=False):
    from dask.dataframe.numeric import to_numeric
    vcounts = data.value_counts(split_every=split_every)
    count_nonzero = vcounts[vcounts != 0]
    count_unique = count_nonzero.size
    stats = [count_unique, data.count(split_every=split_every), vcounts._head(1, npartitions=1, compute=False, safe=False)]
    if is_datetime64_any_dtype(data._meta) and (not datetime_is_numeric):
        min_ts = to_numeric(data.dropna()).min(split_every=split_every)
        max_ts = to_numeric(data.dropna()).max(split_every=split_every)
        stats.extend([min_ts, max_ts])
    stats_names = [(s._name, 0) for s in stats]
    colname = data._meta.name
    name = 'describe-nonnumeric-1d--' + tokenize(data, split_every)
    layer = {(name, 0): (methods.describe_nonnumeric_aggregate, stats_names, colname)}
    graph = HighLevelGraph.from_collections(name, layer, dependencies=stats)
    if not PANDAS_GE_200:
        datetime_is_numeric_kwarg = {'datetime_is_numeric': datetime_is_numeric}
    else:
        datetime_is_numeric_kwarg = {}
    meta = data._meta_nonempty.describe(**datetime_is_numeric_kwarg)
    return new_dd_object(graph, name, meta, divisions=[None, None])