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
def _groupby_aggregate_spec(df, spec, levels=None, dropna=None, sort=False, observed=None, **kwargs):
    """
    A simpler version of _groupby_aggregate that just calls ``aggregate`` using
    the user-provided spec.
    """
    dropna = {'dropna': dropna} if dropna is not None else {}
    observed = {'observed': observed} if observed is not None else {}
    return df.groupby(level=levels, sort=sort, **observed, **dropna).aggregate(spec, **kwargs)