from __future__ import annotations
import re
import warnings
import numpy as np
import pandas as pd
from fsspec.core import expand_paths_if_needed, get_fs_token_paths, stringify_path
from fsspec.spec import AbstractFileSystem
from dask import config
from dask.dataframe.io.utils import _is_local_fs
from dask.utils import natural_sort_key, parse_bytes
def _parse_pandas_metadata(pandas_metadata):
    """Get the set of names from the pandas metadata section

    Parameters
    ----------
    pandas_metadata : dict
        Should conform to the pandas parquet metadata spec

    Returns
    -------
    index_names : list
        List of strings indicating the actual index names
    column_names : list
        List of strings indicating the actual column names
    storage_name_mapping : dict
        Pairs of storage names (e.g. the field names for
        PyArrow) and actual names. The storage and field names will
        differ for index names for certain writers (pyarrow > 0.8).
    column_indexes_names : list
        The names for ``df.columns.name`` or ``df.columns.names`` for
        a MultiIndex in the columns

    Notes
    -----
    This should support metadata written by at least

    * fastparquet>=0.1.3
    * pyarrow>=0.7.0
    """
    index_storage_names = [n['name'] if isinstance(n, dict) else n for n in pandas_metadata['index_columns']]
    index_name_xpr = re.compile('__index_level_\\d+__')
    pairs = [(x.get('field_name', x['name']), x['name']) for x in pandas_metadata['columns']]
    pairs2 = []
    for storage_name, real_name in pairs:
        if real_name and index_name_xpr.match(real_name):
            real_name = None
        pairs2.append((storage_name, real_name))
    index_names = [name for storage_name, name in pairs2 if name != storage_name]
    column_index_names = pandas_metadata.get('column_indexes', [{'name': None}])
    column_index_names = [x['name'] for x in column_index_names]
    if not index_names:
        if index_storage_names and isinstance(index_storage_names[0], dict):
            index_storage_names = []
        index_names = list(index_storage_names)
        index_storage_names2 = set(index_storage_names)
        column_names = [name for storage_name, name in pairs if name not in index_storage_names2]
    else:
        column_names = [name for storage_name, name in pairs2 if name == storage_name]
    storage_name_mapping = dict(pairs2)
    return (index_names, column_names, storage_name_mapping, column_index_names)