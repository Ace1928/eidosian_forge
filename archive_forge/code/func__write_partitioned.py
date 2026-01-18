from __future__ import annotations
import json
import operator
import textwrap
import warnings
from collections import defaultdict
from datetime import datetime
from functools import reduce
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.fs as pa_fs
import pyarrow.parquet as pq
from fsspec.core import expand_paths_if_needed, stringify_path
from fsspec.implementations.arrow import ArrowFSWrapper
from pyarrow import dataset as pa_ds
from pyarrow import fs as pa_fs
import dask
from dask.base import normalize_token, tokenize
from dask.core import flatten
from dask.dataframe._compat import PANDAS_GE_220
from dask.dataframe.backends import pyarrow_schema_dispatch
from dask.dataframe.io.parquet.utils import (
from dask.dataframe.io.utils import _get_pyarrow_dtypes, _is_local_fs, _open_input_files
from dask.dataframe.utils import clear_known_categories, pyarrow_strings_enabled
from dask.delayed import Delayed
from dask.utils import getargspec, natural_sort_key
def _write_partitioned(table, df, root_path, filename, partition_cols, fs, pandas_to_arrow_table, preserve_index, index_cols=(), return_metadata=True, **kwargs):
    """Write table to a partitioned dataset with pyarrow.

    Logic copied from pyarrow.parquet.
    (arrow/python/pyarrow/parquet.py::write_to_dataset)

    TODO: Remove this in favor of pyarrow's `write_to_dataset`
          once ARROW-8244 is addressed.
    """
    fs.mkdirs(root_path, exist_ok=True)
    if preserve_index:
        df.reset_index(inplace=True)
    df = df[table.schema.names]
    index_cols = list(index_cols) if index_cols else []
    preserve_index = False
    if index_cols:
        df.set_index(index_cols, inplace=True)
        preserve_index = True
    partition_keys = [df[col] for col in partition_cols]
    data_df = df.drop(partition_cols, axis='columns')
    data_cols = df.columns.drop(partition_cols)
    if len(data_cols) == 0 and (not index_cols):
        raise ValueError('No data left to save outside partition columns')
    subschema = table.schema
    for col in table.schema.names:
        if col in partition_cols:
            subschema = subschema.remove(subschema.get_field_index(col))
    md_list = []
    partition_keys = partition_keys[0] if len(partition_keys) == 1 else partition_keys
    gb = data_df.groupby(partition_keys, dropna=False, observed=False)
    for keys, subgroup in gb:
        if not isinstance(keys, tuple):
            keys = (keys,)
        subdir = fs.sep.join([_hive_dirname(name, val) for name, val in zip(partition_cols, keys)])
        subtable = pandas_to_arrow_table(subgroup, preserve_index=preserve_index, schema=subschema)
        prefix = fs.sep.join([root_path, subdir])
        fs.mkdirs(prefix, exist_ok=True)
        full_path = fs.sep.join([prefix, filename])
        with fs.open(full_path, 'wb') as f:
            pq.write_table(subtable, f, metadata_collector=md_list if return_metadata else None, **kwargs)
        if return_metadata:
            md_list[-1].set_file_path(fs.sep.join([subdir, filename]))
    return md_list