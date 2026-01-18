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
def _non_agg_chunk(df, *by, key, dropna=None, observed=None, **kwargs):
    """
    A non-aggregation agg function. This simulates the behavior of an initial
    partitionwise aggregation, but doesn't actually aggregate or throw away
    any data.
    """
    if is_series_like(df):
        result = df.to_frame().set_index(by[0] if len(by) == 1 else list(by))[df.name]
    else:
        result = df.set_index(list(by))
        if isinstance(key, (tuple, list, set, pd.Index)):
            key = list(key)
        result = result[key]
    if observed is False:
        has_categoricals = False
        if isinstance(result.index, pd.CategoricalIndex):
            has_categoricals = True
            full_index = result.index.categories.copy().rename(result.index.name)
        elif isinstance(result.index, pd.MultiIndex) and any((isinstance(level, pd.CategoricalIndex) for level in result.index.levels)):
            has_categoricals = True
            full_index = pd.MultiIndex.from_product(result.index.levels, names=result.index.names)
        if has_categoricals:
            new_cats = full_index[~full_index.isin(result.index)]
            empty_data = {c: pd.Series(index=new_cats, dtype=result[c].dtype) for c in result.columns}
            empty = pd.DataFrame(empty_data)
            result = pd.concat([result, empty])
    return result