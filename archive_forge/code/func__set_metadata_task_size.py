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
def _set_metadata_task_size(metadata_task_size, fs):
    if metadata_task_size is None:
        config_str = 'dataframe.parquet.metadata-task-size-' + ('local' if _is_local_fs(fs) else 'remote')
        return config.get(config_str, 0)
    return metadata_task_size