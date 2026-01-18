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
def _cov_corr_chunk(df, corr=False):
    """Chunk part of a covariance or correlation computation"""
    shape = (df.shape[1], df.shape[1])
    kwargs = {} if PANDAS_GE_300 else {'copy': False}
    df = df.astype('float64', **kwargs)
    sums = np.zeros_like(df.values, shape=shape)
    counts = np.zeros_like(df.values, shape=shape)
    for idx in range(len(df.columns)):
        mask = df.iloc[:, idx].notnull()
        sums[idx] = df[mask].sum().values
        counts[idx] = df[mask].count().values
    if df.shape[0] == 1:
        cov = np.full_like(sums, np.nan)
    else:
        cov = df.cov().values
    dtype = [('sum', sums.dtype), ('count', counts.dtype), ('cov', cov.dtype)]
    if corr:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            mu = (sums / counts).T
        m = np.zeros_like(df.values, shape=shape)
        mask = df.isnull().values
        for idx in range(len(df.columns)):
            mu_discrepancy = np.subtract(df.iloc[:, idx].values[:, None], mu[idx][None, :]) ** 2
            mu_discrepancy[mask] = np.nan
            m[idx] = np.nansum(mu_discrepancy, axis=0)
        m = m.T
        dtype.append(('m', m.dtype))
    out = {'sum': sums, 'count': counts, 'cov': cov * (counts - 1)}
    if corr:
        out['m'] = m
    return out