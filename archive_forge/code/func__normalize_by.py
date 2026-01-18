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
def _normalize_by(df, by):
    """Replace series with column names wherever possible."""
    if not isinstance(df, DataFrame):
        return by
    elif isinstance(by, list):
        return [_normalize_by(df, col) for col in by]
    elif is_series_like(by) and by.name in df.columns and (by._name == df[by.name]._name):
        return by.name
    elif isinstance(by, DataFrame) and set(by.columns).issubset(df.columns) and (by._name == df[by.columns]._name):
        return list(by.columns)
    else:
        return by