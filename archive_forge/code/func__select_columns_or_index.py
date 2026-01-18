from __future__ import annotations
import math
import pickle
import warnings
from functools import partial, wraps
import numpy as np
import pandas as pd
from pandas.api.types import is_dtype_equal
from tlz import merge_sorted, unique
from dask.base import is_dask_collection, tokenize
from dask.dataframe import methods
from dask.dataframe.core import (
from dask.dataframe.dispatch import group_split_dispatch, hash_object_dispatch
from dask.dataframe.io import from_pandas
from dask.dataframe.shuffle import (
from dask.dataframe.utils import (
from dask.highlevelgraph import HighLevelGraph
from dask.layers import BroadcastJoinLayer
from dask.utils import M, apply, get_default_shuffle_method
def _select_columns_or_index(df, columns_or_index):
    """
    Returns a DataFrame with columns corresponding to each
    column or index level in columns_or_index.  If included,
    the column corresponding to the index level is named _index.

    This is the local (non-collection) version of
    ``dask.core.DataFrame._select_columns_or_index``.
    """

    def _is_column_label_reference(df, key):
        return (np.isscalar(key) or isinstance(key, tuple)) and key in df.columns
    columns_or_index = columns_or_index if isinstance(columns_or_index, list) else [columns_or_index]
    column_names = [n for n in columns_or_index if _is_column_label_reference(df, n)]
    selected_df = df[column_names]
    if _contains_index_name(df, columns_or_index):
        selected_df = selected_df.assign(_index=df.index)
    return selected_df