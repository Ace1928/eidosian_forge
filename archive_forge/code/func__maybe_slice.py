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
def _maybe_slice(grouped, columns):
    """
    Slice columns if grouped is pd.DataFrameGroupBy
    """
    if 'groupby' in type(grouped).__name__.lower():
        if columns is not None:
            if isinstance(columns, (tuple, list, set, pd.Index)):
                columns = list(columns)
            return grouped[columns]
    return grouped