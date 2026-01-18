from __future__ import annotations
import json
from typing import Protocol, runtime_checkable
from uuid import uuid4
import fsspec
import pandas as pd
from fsspec.implementations.local import LocalFileSystem
from packaging.version import parse as parse_version
def _meta_from_dtypes(to_read_columns, file_dtypes, index_cols, column_index_names):
    """Get the final metadata for the dask.dataframe

    Parameters
    ----------
    to_read_columns : list
        All the columns to end up with, including index names
    file_dtypes : dict
        Mapping from column name to dtype for every element
        of ``to_read_columns``
    index_cols : list
        Subset of ``to_read_columns`` that should move to the
        index
    column_index_names : list
        The values for df.columns.name for a MultiIndex in the
        columns, or df.index.name for a regular Index in the columns

    Returns
    -------
    meta : DataFrame
    """
    data = {c: pd.Series([], dtype=file_dtypes.get(c, 'int64')) for c in to_read_columns}
    indexes = [data.pop(c) for c in index_cols or []]
    if len(indexes) == 0:
        index = None
    elif len(index_cols) == 1:
        index = indexes[0]
        if index_cols[0] != '__index_level_0__':
            index.name = index_cols[0]
    else:
        index = pd.MultiIndex.from_arrays(indexes, names=index_cols)
    df = pd.DataFrame(data, index=index)
    if column_index_names:
        df.columns.names = column_index_names
    return df