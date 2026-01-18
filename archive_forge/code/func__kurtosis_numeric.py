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
def _kurtosis_numeric(self, fisher=True, bias=True, nan_policy='propagate'):
    """Method for dataframes with numeric columns.

        Maps the array version from da.stats onto the numeric array of columns.
        """
    from dask.array import stats as da_stats
    num = self.select_dtypes(include=['number', 'bool'], exclude=[np.timedelta64])
    values_dtype = num.values.dtype
    array_values = num.values
    if not np.issubdtype(values_dtype, np.number):
        array_values = num.values.astype('f8')
    array_kurtosis = da_stats.kurtosis(array_values, axis=0, fisher=fisher, bias=bias, nan_policy=nan_policy)
    name = self._token_prefix + 'kurtosis-numeric' + tokenize(num)
    cols = num._meta.columns if is_dataframe_like(num) else None
    with check_reductions_runtime_warning():
        kurtosis_shape = num._meta_nonempty.var(axis=0).shape
    array_kurtosis_name = (array_kurtosis._name,) + (0,) * len(kurtosis_shape)
    layer = {(name, 0): (methods.wrap_kurtosis_reduction, array_kurtosis_name, cols)}
    graph = HighLevelGraph.from_collections(name, layer, dependencies=[array_kurtosis])
    return new_dd_object(graph, name, num._meta_nonempty.kurtosis(), divisions=[None, None])