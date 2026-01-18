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
def _normalize_spec(spec, non_group_columns):
    """
    Return a list of ``(result_column, func, input_column)`` tuples.

    Spec can be

    - a function
    - a list of functions
    - a dictionary that maps input-columns to functions
    - a dictionary that maps input-columns to a lists of functions
    - a dictionary that maps input-columns to a dictionaries that map
      output-columns to functions.

    The non-group columns are a list of all column names that are not used in
    the groupby operation.

    Usually, the result columns are mutli-level names, returned as tuples.
    If only a single function is supplied or dictionary mapping columns
    to single functions, simple names are returned as strings (see the first
    two examples below).

    Examples
    --------
    >>> _normalize_spec('mean', ['a', 'b', 'c'])
    [('a', 'mean', 'a'), ('b', 'mean', 'b'), ('c', 'mean', 'c')]

    >>> spec = collections.OrderedDict([('a', 'mean'), ('b', 'count')])
    >>> _normalize_spec(spec, ['a', 'b', 'c'])
    [('a', 'mean', 'a'), ('b', 'count', 'b')]

    >>> _normalize_spec(['var', 'mean'], ['a', 'b', 'c'])
    ... # doctest: +NORMALIZE_WHITESPACE
    [(('a', 'var'), 'var', 'a'), (('a', 'mean'), 'mean', 'a'),      (('b', 'var'), 'var', 'b'), (('b', 'mean'), 'mean', 'b'),      (('c', 'var'), 'var', 'c'), (('c', 'mean'), 'mean', 'c')]

    >>> spec = collections.OrderedDict([('a', 'mean'), ('b', ['sum', 'count'])])
    >>> _normalize_spec(spec, ['a', 'b', 'c'])
    ... # doctest: +NORMALIZE_WHITESPACE
    [(('a', 'mean'), 'mean', 'a'), (('b', 'sum'), 'sum', 'b'),       (('b', 'count'), 'count', 'b')]

    >>> spec = collections.OrderedDict()
    >>> spec['a'] = ['mean', 'size']
    >>> spec['b'] = collections.OrderedDict([('e', 'count'), ('f', 'var')])
    >>> _normalize_spec(spec, ['a', 'b', 'c'])
    ... # doctest: +NORMALIZE_WHITESPACE
    [(('a', 'mean'), 'mean', 'a'), (('a', 'size'), 'size', 'a'),      (('b', 'e'), 'count', 'b'), (('b', 'f'), 'var', 'b')]
    """
    if not isinstance(spec, dict):
        spec = collections.OrderedDict(zip(non_group_columns, it.repeat(spec)))
    res = []
    if isinstance(spec, dict):
        for input_column, subspec in spec.items():
            if isinstance(subspec, dict):
                res.extend((((input_column, result_column), func, input_column) for result_column, func in subspec.items()))
            else:
                if not isinstance(subspec, list):
                    subspec = [subspec]
                res.extend((((input_column, funcname(func)), func, input_column) for func in subspec))
    else:
        raise ValueError(f'unsupported agg spec of type {type(spec)}')
    compounds = (list, tuple, dict)
    use_flat_columns = not any((isinstance(subspec, compounds) for subspec in spec.values()))
    if use_flat_columns:
        res = [(input_col, func, input_col) for _, func, input_col in res]
    return res