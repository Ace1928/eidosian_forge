from __future__ import annotations
import collections
import itertools as it
import operator
import uuid
import warnings
from functools import partial, wraps
from numbers import Integral
import numpy as np
import pandas as pd
from dask.base import is_dask_collection, tokenize
from dask.core import flatten
from dask.dataframe._compat import (
from dask.dataframe.core import (
from dask.dataframe.dispatch import grouper_dispatch
from dask.dataframe.methods import concat, drop_columns
from dask.dataframe.utils import (
from dask.highlevelgraph import HighLevelGraph
from dask.typing import no_default
from dask.utils import (
def _build_agg_args_single(result_column, func, func_args, func_kwargs, input_column):
    simple_impl = {'sum': (M.sum, M.sum), 'min': (M.min, M.min), 'max': (M.max, M.max), 'count': (M.count, M.sum), 'size': (M.size, M.sum), 'first': (M.first, M.first), 'last': (M.last, M.last), 'prod': (M.prod, M.prod), 'median': (None, M.median)}
    if func in simple_impl.keys():
        return _build_agg_args_simple(result_column, func, input_column, simple_impl[func])
    elif func == 'var':
        return _build_agg_args_var(result_column, func, func_args, func_kwargs, input_column)
    elif func == 'std':
        return _build_agg_args_std(result_column, func, func_args, func_kwargs, input_column)
    elif func == 'mean':
        return _build_agg_args_mean(result_column, func, input_column)
    elif func == 'list':
        return _build_agg_args_list(result_column, func, input_column)
    elif isinstance(func, Aggregation):
        return _build_agg_args_custom(result_column, func, input_column)
    else:
        raise ValueError(f'unknown aggregate {func}')