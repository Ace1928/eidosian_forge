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
def _split_user_options(**kwargs):
    user_kwargs = kwargs.copy()
    if 'file' in user_kwargs:
        warnings.warn("Passing user options with the 'file' argument is now deprecated. Please use 'dataset' instead.", FutureWarning)
    dataset_options = {**user_kwargs.pop('file', {}).copy(), **user_kwargs.pop('dataset', {}).copy()}
    read_options = user_kwargs.pop('read', {}).copy()
    open_file_options = user_kwargs.pop('open_file_options', {}).copy()
    return (dataset_options, read_options, open_file_options, user_kwargs)