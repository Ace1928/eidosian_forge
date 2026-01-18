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
def _one_path_one_key(path, key, start, stop, chunksize):
    """
    Get the DataFrame corresponding to one path and one key (which
    should not contain any wildcards).
    """
    if start >= stop:
        raise ValueError('Start row number ({}) is above or equal to stop row number ({})'.format(start, stop))
    return [(path, key, {'start': s, 'stop': s + chunksize}) for i, s in enumerate(range(start, stop, chunksize))]