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
def _process_open_file_options(open_file_options, metadata=None, columns=None, row_groups=None, default_engine=None, default_cache='readahead', allow_precache=True):
    open_file_options = (open_file_options or {}).copy()
    precache_options = open_file_options.pop('precache_options', {}).copy()
    if not allow_precache:
        precache_options = {}
    if 'open_file_func' not in open_file_options:
        if precache_options.get('method', None) == 'parquet':
            open_file_options['cache_type'] = open_file_options.get('cache_type', 'parts')
            precache_options.update({'metadata': metadata, 'columns': columns, 'row_groups': row_groups, 'engine': precache_options.get('engine', default_engine)})
        else:
            open_file_options['cache_type'] = open_file_options.get('cache_type', default_cache)
            open_file_options['mode'] = open_file_options.get('mode', 'rb')
    return (precache_options, open_file_options)