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
class ToParquetFunctionWrapper:
    """
    Parquet Function-Wrapper Class

    Writes a DataFrame partition into a distinct parquet
    file. When called, the function also requires the
    current block index (via ``blockwise.BlockIndex``).
    """

    def __init__(self, engine, path, fs, partition_on, write_metadata_file, i_offset, name_function, kwargs_pass):
        self.engine = engine
        self.path = path
        self.fs = fs
        self.partition_on = partition_on
        self.write_metadata_file = write_metadata_file
        self.i_offset = i_offset
        self.name_function = name_function
        self.kwargs_pass = kwargs_pass
        self.__name__ = 'to-parquet'

    def __dask_tokenize__(self):
        return (self.engine, self.path, self.fs, self.partition_on, self.write_metadata_file, self.i_offset, self.name_function, self.kwargs_pass)

    def __call__(self, df, block_index: tuple[int]):
        part_i = block_index[0]
        filename = f'part.{part_i + self.i_offset}.parquet' if self.name_function is None else self.name_function(part_i + self.i_offset)
        return self.engine.write_partition(df, self.path, self.fs, filename, self.partition_on, self.write_metadata_file, **dict(self.kwargs_pass, head=True) if part_i == 0 else self.kwargs_pass)