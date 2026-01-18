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
def _drop_duplicates_shuffle(self, split_out, split_every, shuffle_method, ignore_index, **kwargs):
    if isinstance(self, Index):
        df = self.to_frame(name=self.name or '__index__')
    elif isinstance(self, Series):
        df = self.to_frame(name=self.name or '__series__')
    else:
        df = self
    split_every = 8 if split_every is None else split_every
    shuffle_npartitions = max(df.npartitions // (split_every or df.npartitions), split_out)
    chunk = M.drop_duplicates
    deduplicated = df.map_partitions(chunk, token='drop-duplicates-chunk', meta=df._meta, ignore_index=ignore_index, enforce_metadata=False, transform_divisions=False, **kwargs).shuffle(kwargs.get('subset', None) or list(df.columns), ignore_index=ignore_index, npartitions=shuffle_npartitions, shuffle_method=shuffle_method).map_partitions(chunk, meta=df._meta, ignore_index=ignore_index, token='drop-duplicates-agg', transform_divisions=False, **kwargs)
    if isinstance(self, Index):
        deduplicated = deduplicated.set_index(self.name or '__index__', sort=False).index
        if deduplicated.name == '__index__':
            deduplicated.name = None
    elif isinstance(self, Series):
        deduplicated = deduplicated[self.name or '__series__']
        if deduplicated.name == '__series__':
            deduplicated.name = None
    return deduplicated.repartition(npartitions=split_out)