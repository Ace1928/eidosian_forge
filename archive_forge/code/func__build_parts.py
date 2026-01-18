from __future__ import annotations
import os
import uuid
from fnmatch import fnmatch
from glob import glob
from warnings import warn
import pandas as pd
from fsspec.utils import build_name_function, stringify_path
from tlz import merge
from dask import config
from dask.base import (
from dask.dataframe.backends import dataframe_creation_dispatch
from dask.dataframe.core import DataFrame, Scalar
from dask.dataframe.io.io import _link, from_map
from dask.dataframe.io.utils import DataFrameIOFunction, SupportsLock
from dask.highlevelgraph import HighLevelGraph
from dask.utils import get_scheduler_lock
from dask.dataframe.core import _Frame
def _build_parts(paths, key, start, stop, chunksize, sorted_index, mode):
    """
    Build the list of partition inputs and divisions for read_hdf
    """
    parts = []
    global_divisions = []
    for path in paths:
        keys, stops, divisions = _get_keys_stops_divisions(path, key, stop, sorted_index, chunksize, mode)
        for k, s, d in zip(keys, stops, divisions):
            if d and global_divisions:
                global_divisions = global_divisions[:-1] + d
            elif d:
                global_divisions = d
            parts.extend(_one_path_one_key(path, k, start, s, chunksize))
    return (parts, global_divisions or [None] * (len(parts) + 1))