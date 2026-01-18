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
@classmethod
def _bind_operator_method(cls, name, op, original=pd.DataFrame):
    """bind operator method like DataFrame.add to this class"""

    def meth(self, other, axis='columns', level=None, fill_value=None):
        if level is not None:
            raise NotImplementedError('level must be None')
        axis = self._validate_axis(axis)
        if axis in (1, 'columns'):
            if isinstance(other, Series):
                msg = f'Unable to {name} dd.Series with axis=1'
                raise ValueError(msg)
            elif is_series_like(other):
                meta = _emulate(op, self, other=other, axis=axis, fill_value=fill_value)
                return map_partitions(op, self, other=other, meta=meta, axis=axis, fill_value=fill_value, enforce_metadata=False)
        meta = _emulate(op, self, other, axis=axis, fill_value=fill_value)
        return map_partitions(op, self, other, meta=meta, axis=axis, fill_value=fill_value, enforce_metadata=False)
    meth.__name__ = name
    setattr(cls, name, derived_from(original)(meth))