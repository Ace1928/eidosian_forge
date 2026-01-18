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
def _sort_and_analyze_paths(file_list, fs, root=False):
    file_list = sorted(file_list, key=natural_sort_key)
    base, fns = _analyze_paths(file_list, fs, root=root)
    return (file_list, base, fns)