from __future__ import annotations
import contextlib
import math
import warnings
from typing import Literal
import pandas as pd
import tlz as toolz
from fsspec.core import get_fs_token_paths
from fsspec.utils import stringify_path
import dask
from dask.base import tokenize
from dask.blockwise import BlockIndex
from dask.dataframe.backends import dataframe_creation_dispatch
from dask.dataframe.core import DataFrame, Scalar
from dask.dataframe.io.io import from_map
from dask.dataframe.io.parquet.utils import (
from dask.dataframe.io.utils import DataFrameIOFunction, _is_local_fs
from dask.dataframe.methods import concat
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph
from dask.layers import DataFrameIOLayer
from dask.utils import apply, import_required, natural_sort_key, parse_bytes
class ParquetFunctionWrapper(DataFrameIOFunction):
    """
    Parquet Function-Wrapper Class
    Reads parquet data from disk to produce a partition
    (given a `part` argument).
    """

    def __init__(self, engine, fs, meta, columns, index, dtype_backend, kwargs, common_kwargs):
        self.engine = engine
        self.fs = fs
        self.meta = meta
        self._columns = columns
        self.index = index
        self.dtype_backend = dtype_backend
        self.common_kwargs = toolz.merge(common_kwargs, kwargs or {})

    @property
    def columns(self):
        return self._columns

    def project_columns(self, columns):
        """Return a new ParquetFunctionWrapper object
        with a sub-column projection.
        """
        if columns == self.columns:
            return self
        return ParquetFunctionWrapper(self.engine, self.fs, self.meta, columns, self.index, self.dtype_backend, None, self.common_kwargs)

    def __call__(self, part):
        if not isinstance(part, list):
            part = [part]
        return read_parquet_part(self.fs, self.engine, self.meta, [(p.data['piece'], p.data.get('kwargs', {})) if hasattr(p, 'data') else (p['piece'], p.get('kwargs', {})) for p in part], self.columns, self.index, self.common_kwargs)