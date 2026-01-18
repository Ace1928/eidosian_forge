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
def _var_agg(g, levels, ddof, sort=False, numeric_only=no_default, observed=False, dropna=True):
    numeric_only_kwargs = get_numeric_only_kwargs(numeric_only)
    g = g.groupby(level=levels, sort=sort, observed=observed, dropna=dropna).sum(**numeric_only_kwargs)
    nc = len(g.columns)
    x = g[g.columns[:nc // 3]]
    x2 = g[g.columns[nc // 3:2 * nc // 3]].rename(columns=lambda c: c[0])
    n = g[g.columns[-nc // 3:]].rename(columns=lambda c: c[0])
    result = x2 - x ** 2 / n
    div = n - ddof
    div[div < 0] = 0
    result /= div
    result[n - ddof == 0] = np.nan
    assert is_dataframe_like(result)
    result[result < 0] = 0
    return result