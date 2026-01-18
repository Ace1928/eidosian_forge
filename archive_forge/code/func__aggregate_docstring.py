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
def _aggregate_docstring(based_on=None):
    based_on_str = '\n' if based_on is None else f'\nBased on {based_on}\n'

    def wrapper(func):
        func.__doc__ = f"""Aggregate using one or more specified operations\n        {based_on_str}\n        Parameters\n        ----------\n        arg : callable, str, list or dict, optional\n            Aggregation spec. Accepted combinations are:\n\n            - callable function\n            - string function name\n            - list of functions and/or function names, e.g. ``[np.sum, 'mean']``\n            - dict of column names -> function, function name or list of such.\n            - None only if named aggregation syntax is used\n        split_every : int, optional\n            Number of intermediate partitions that may be aggregated at once.\n            This defaults to 8. If your intermediate partitions are likely to\n            be small (either due to a small number of groups or a small initial\n            partition size), consider increasing this number for better performance.\n        split_out : int, optional\n            Number of output partitions. Default is 1.\n        shuffle : bool or str, optional\n            Whether a shuffle-based algorithm should be used. A specific\n            algorithm name may also be specified (e.g. ``"tasks"`` or ``"p2p"``).\n            The shuffle-based algorithm is likely to be more efficient than\n            ``shuffle=False`` when ``split_out>1`` and the number of unique\n            groups is large (high cardinality). Default is ``False`` when\n            ``split_out = 1``. When ``split_out > 1``, it chooses the algorithm\n            set by the ``shuffle`` option in the dask config system, or ``"tasks"``\n            if nothing is set.\n        kwargs: tuple or pd.NamedAgg, optional\n            Used for named aggregations where the keywords are the output column\n            names and the values are tuples where the first element is the input\n            column name and the second element is the aggregation function.\n            ``pandas.NamedAgg`` can also be used as the value. To use the named\n            aggregation syntax, arg must be set to None.\n        """
        return func
    return wrapper