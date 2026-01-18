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
def _apply_chunk(df, *by, dropna=None, observed=None, **kwargs):
    func = kwargs.pop('chunk')
    columns = kwargs.pop('columns')
    dropna = {'dropna': dropna} if dropna is not None else {}
    observed = {'observed': observed} if observed is not None else {}
    g = _groupby_raise_unaligned(df, by=by, **observed, **dropna)
    if is_series_like(df) or columns is None:
        return func(g, **kwargs)
    else:
        if isinstance(columns, (tuple, list, set, pd.Index)):
            columns = list(columns)
        return func(g[columns], **kwargs)