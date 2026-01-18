from __future__ import annotations
import os
from collections.abc import Mapping
from io import BytesIO
from warnings import catch_warnings, simplefilter, warn
import numpy as np
import pandas as pd
from fsspec.compression import compr
from fsspec.core import get_fs_token_paths
from fsspec.core import open as open_file
from fsspec.core import open_files
from fsspec.utils import infer_compression
from pandas.api.types import (
from dask.base import tokenize
from dask.bytes import read_bytes
from dask.core import flatten
from dask.dataframe.backends import dataframe_creation_dispatch
from dask.dataframe.io.io import from_map
from dask.dataframe.io.utils import DataFrameIOFunction
from dask.dataframe.utils import clear_known_categories
from dask.delayed import delayed
from dask.utils import asciitable, parse_bytes
from the start of the file (or of the first file if it's a glob). Usually this
from dask.dataframe.core import _Frame
def _write_csv(df, fil, *, depend_on=None, **kwargs):
    with fil as f:
        df.to_csv(f, **kwargs)
    return os.path.normpath(fil.path)