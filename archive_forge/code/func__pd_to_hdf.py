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
def _pd_to_hdf(pd_to_hdf, lock, args, kwargs=None):
    """A wrapper function around pd_to_hdf that enables locking"""
    if lock:
        lock.acquire()
    try:
        pd_to_hdf(*args, **kwargs)
    finally:
        if lock:
            lock.release()
    return None