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
def _get_keys_stops_divisions(path, key, stop, sorted_index, chunksize, mode):
    """
    Get the "keys" or group identifiers which match the given key, which
    can contain wildcards (see _expand_path). This uses the hdf file
    identified by the given path. Also get the index of the last row of
    data for each matched key.
    """
    with pd.HDFStore(path, mode=mode) as hdf:
        stops = []
        divisions = []
        keys = _expand_key(key, hdf)
        for k in keys:
            storer = hdf.get_storer(k)
            if storer.format_type != 'table':
                raise TypeError(dont_use_fixed_error_message)
            if stop is None:
                stops.append(storer.nrows)
            elif stop > storer.nrows:
                raise ValueError('Stop keyword exceeds dataset number of rows ({})'.format(storer.nrows))
            else:
                stops.append(stop)
            if sorted_index:
                division = [storer.read_column('index', start=start, stop=start + 1)[0] for start in range(0, storer.nrows, chunksize)]
                division_end = storer.read_column('index', start=storer.nrows - 1, stop=storer.nrows)[0]
                division.append(division_end)
                divisions.append(division)
            else:
                divisions.append(None)
    return (keys, stops, divisions)