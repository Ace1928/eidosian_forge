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
def _repartition_from_boundaries(df, new_partitions_boundaries, new_name):
    if not isinstance(new_partitions_boundaries, list):
        new_partitions_boundaries = list(new_partitions_boundaries)
    if new_partitions_boundaries[0] > 0:
        new_partitions_boundaries.insert(0, 0)
    if new_partitions_boundaries[-1] < df.npartitions:
        new_partitions_boundaries.append(df.npartitions)
    dsk = {}
    for i, (start, end) in enumerate(zip(new_partitions_boundaries, new_partitions_boundaries[1:])):
        dsk[new_name, i] = (methods.concat, [(df._name, j) for j in range(start, end)])
    divisions = [df.divisions[i] for i in new_partitions_boundaries]
    graph = HighLevelGraph.from_collections(new_name, dsk, dependencies=[df])
    return new_dd_object(graph, new_name, df._meta, divisions)