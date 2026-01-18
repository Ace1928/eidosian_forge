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
def _describe_1d(self, data, split_every=False, percentiles=None, percentiles_method='default', datetime_is_numeric=False):
    if is_bool_dtype(data._meta):
        return self._describe_nonnumeric_1d(data, split_every=split_every, datetime_is_numeric=datetime_is_numeric)
    elif is_numeric_dtype(data._meta):
        return self._describe_numeric(data, split_every=split_every, percentiles=percentiles, percentiles_method=percentiles_method)
    elif is_timedelta64_dtype(data._meta):
        return self._describe_numeric(data.dropna(), split_every=split_every, percentiles=percentiles, percentiles_method=percentiles_method, is_timedelta_column=True)
    elif is_datetime64_any_dtype(data._meta) and datetime_is_numeric:
        return self._describe_numeric(data.dropna(), split_every=split_every, percentiles=percentiles, percentiles_method=percentiles_method, is_datetime_column=True)
    else:
        return self._describe_nonnumeric_1d(data, split_every=split_every, datetime_is_numeric=datetime_is_numeric)