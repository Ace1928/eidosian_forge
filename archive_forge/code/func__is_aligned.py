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
def _is_aligned(df, by):
    """Check if ``df`` and ``by`` have aligned indices"""
    if is_series_like(by) or is_dataframe_like(by):
        return df.index.equals(by.index)
    elif isinstance(by, (list, tuple)):
        return all((_is_aligned(df, i) for i in by))
    else:
        return True