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
def _get_divisions_map_partitions(align_dataframes, transform_divisions, dfs, func, args, kwargs):
    """
    Helper to get divisions for map_partitions and map_overlap output.
    """
    if align_dataframes:
        divisions = dfs[0].divisions
    else:
        divisions = max((d.divisions for d in dfs), key=len)
    if transform_divisions and isinstance(dfs[0], Index) and (len(dfs) == 1):
        try:
            divisions = func(*[pd.Index(a.divisions) if a is dfs[0] else a for a in args], **kwargs)
            if isinstance(divisions, pd.Index):
                divisions = methods.tolist(divisions)
        except Exception:
            pass
        else:
            if not valid_divisions(divisions):
                divisions = [None] * (dfs[0].npartitions + 1)
    return divisions