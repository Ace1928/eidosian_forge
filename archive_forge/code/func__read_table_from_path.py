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
def _read_table_from_path(path, fs, row_groups, columns, schema, filters, **kwargs):
    """Read arrow table from file path.

    Used by `ArrowDatasetEngine._read_table` when no filters
    are specified (otherwise fragments are converted directly
    into tables).
    """
    read_kwargs = kwargs.get('read', {}).copy()
    precache_options, open_file_options = _process_open_file_options(read_kwargs.pop('open_file_options', {}), **{'allow_precache': False, 'default_cache': 'none'} if _is_local_fs(fs) else {'columns': columns, 'row_groups': row_groups if row_groups == [None] else [row_groups], 'default_engine': 'pyarrow', 'default_cache': 'none'})
    pre_buffer_default = precache_options.get('method', None) is None
    pre_buffer = {'pre_buffer': read_kwargs.pop('pre_buffer', pre_buffer_default)}
    with _open_input_files([path], fs=fs, precache_options=precache_options, **open_file_options)[0] as fil:
        if row_groups == [None]:
            return pq.ParquetFile(fil, **pre_buffer).read(columns=columns, use_threads=False, use_pandas_metadata=True, **read_kwargs)
        else:
            return pq.ParquetFile(fil, **pre_buffer).read_row_groups(row_groups, columns=columns, use_threads=False, use_pandas_metadata=True, **read_kwargs)