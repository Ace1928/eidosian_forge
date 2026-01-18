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
def _extract_meta(x, nonempty=False):
    """
    Extract internal cache data (``_meta``) from dd.DataFrame / dd.Series
    """
    if isinstance(x, (Scalar, _Frame)):
        return x._meta_nonempty if nonempty else x._meta
    elif isinstance(x, list):
        return [_extract_meta(_x, nonempty) for _x in x]
    elif isinstance(x, tuple):
        return tuple((_extract_meta(_x, nonempty) for _x in x))
    elif isinstance(x, dict):
        res = {}
        for k in x:
            res[k] = _extract_meta(x[k], nonempty)
        return res
    elif isinstance(x, Delayed):
        raise ValueError('Cannot infer dataframe metadata with a `dask.delayed` argument')
    else:
        return x