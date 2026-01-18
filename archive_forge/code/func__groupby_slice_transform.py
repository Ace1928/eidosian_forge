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
def _groupby_slice_transform(df, grouper, key, func, *args, group_keys=GROUP_KEYS_DEFAULT, dropna=None, observed=None, **kwargs):
    dropna = {'dropna': dropna} if dropna is not None else {}
    observed = {'observed': observed} if observed is not None else {}
    g = df.groupby(grouper, group_keys=group_keys, **observed, **dropna)
    if key:
        g = g[key]
    if len(df) == 0:
        return g.apply(func, *args, **kwargs)
    return g.transform(func, *args, **kwargs)