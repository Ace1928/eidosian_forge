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
def _set_gather_statistics(gather_statistics, blocksize, split_row_groups, aggregation_depth, filter_columns, stat_columns):
    if blocksize and split_row_groups is True or (int(split_row_groups) > 1 and aggregation_depth) or filter_columns.intersection(stat_columns):
        gather_statistics = True
    elif not stat_columns:
        gather_statistics = False
    return bool(gather_statistics)