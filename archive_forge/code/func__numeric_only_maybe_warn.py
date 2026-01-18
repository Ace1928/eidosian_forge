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
def _numeric_only_maybe_warn(df, numeric_only, default=None):
    """Update numeric_only to get rid of no_default, and possibly warn about default value.
    TODO: should move to numeric_only decorator. See https://github.com/dask/dask/pull/9952
    """
    if is_dataframe_like(df):
        warn_numeric_only = False
        if numeric_only is no_default:
            if PANDAS_GE_200:
                numeric_only = False
            else:
                warn_numeric_only = True
        numerics = df._meta._get_numeric_data()
        has_non_numerics = len(numerics.columns) < len(df._meta.columns)
        if has_non_numerics:
            if numeric_only is False:
                raise NotImplementedError("'numeric_only=False' is not implemented in Dask.")
            elif warn_numeric_only:
                warnings.warn('The default value of numeric_only in dask will be changed to False in the future when using dask with pandas 2.0', FutureWarning)
    if numeric_only is no_default and default is not None:
        numeric_only = default
    return {} if numeric_only is no_default else {'numeric_only': numeric_only}