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
def _reduction_agg(self, name, axis=None, skipna=True, split_every=False, out=None, numeric_only=None, none_is_zero=True):
    axis = self._validate_axis(axis, none_is_zero=none_is_zero)
    if has_keyword(getattr(self._meta_nonempty, name), 'numeric_only'):
        numeric_only_kwargs = {'numeric_only': numeric_only}
    else:
        numeric_only_kwargs = {}
    with check_numeric_only_deprecation(name, True):
        meta = getattr(self._meta_nonempty, name)(axis=axis, skipna=skipna, **numeric_only_kwargs)
    token = self._token_prefix + name
    if axis == 1:
        result = self.map_partitions(_getattr_numeric_only, meta=meta, token=token, skipna=skipna, axis=axis, _dask_method_name=name, **numeric_only_kwargs)
        return handle_out(out, result)
    else:
        result = self.reduction(_getattr_numeric_only, meta=meta, token=token, skipna=skipna, axis=axis, split_every=split_every, _dask_method_name=name, **numeric_only_kwargs)
        if isinstance(self, DataFrame) and isinstance(result, Series):
            result.divisions = (self.columns.min(), self.columns.max())
        return handle_out(out, result)