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
def _partitions(self, index):
    if not isinstance(index, tuple):
        index = (index,)
    from dask.array.slicing import normalize_index
    index = normalize_index(index, (self.npartitions,))
    index = tuple((slice(k, k + 1) if isinstance(k, Number) else k for k in index))
    name = 'blocks-' + tokenize(self, index)
    new_keys = np.array(self.__dask_keys__(), dtype=object)[index].tolist()
    divisions = [self.divisions[i] for _, i in new_keys] + [self.divisions[new_keys[-1][1] + 1]]
    dsk = {(name, i): tuple(key) for i, key in enumerate(new_keys)}
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[self])
    return new_dd_object(graph, name, self._meta, divisions)