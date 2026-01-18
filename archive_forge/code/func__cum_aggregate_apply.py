from __future__ import annotations
import warnings
from functools import partial
import numpy as np
import pandas as pd
from pandas.api.types import is_extension_array_dtype
from pandas.errors import PerformanceWarning
from tlz import partition
from dask.dataframe._compat import (
from dask.dataframe.dispatch import (  # noqa: F401
from dask.dataframe.utils import is_dataframe_like, is_index_like, is_series_like
from dask.utils import _deprecated_kwarg
def _cum_aggregate_apply(aggregate, x, y):
    """Apply aggregation function within a cumulative aggregation

    Parameters
    ----------
    aggregate: function (a, a) -> a
        The aggregation function, like add, which is used to and subsequent
        results
    x:
    y:
    """
    if y is None:
        return x
    else:
        return aggregate(x, y)