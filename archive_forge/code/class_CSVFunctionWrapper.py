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
class CSVFunctionWrapper(DataFrameIOFunction):
    """
    CSV Function-Wrapper Class
    Reads CSV data from disk to produce a partition (given a key).
    """

    def __init__(self, full_columns, columns, colname, head, header, reader, dtypes, enforce, kwargs):
        self.full_columns = full_columns
        self._columns = columns
        self.colname = colname
        self.head = head
        self.header = header
        self.reader = reader
        self.dtypes = dtypes
        self.enforce = enforce
        self.kwargs = kwargs

    @property
    def columns(self):
        if self._columns is None:
            return self.full_columns
        if self.colname:
            return self._columns + [self.colname]
        return self._columns

    def project_columns(self, columns):
        """Return a new CSVFunctionWrapper object with
        a sub-column projection.
        """
        columns = [c for c in self.head.columns if c in columns]
        if columns == self.columns:
            return self
        if self.colname and self.colname not in columns:
            head = self.head[columns + [self.colname]]
        else:
            head = self.head[columns]
        return CSVFunctionWrapper(self.full_columns, columns, self.colname, head, self.header, self.reader, {c: self.dtypes[c] for c in columns}, self.enforce, self.kwargs)

    def __call__(self, part):
        block, path, is_first, is_last = part
        if path is not None:
            path_info = (self.colname, path, sorted(list(self.head[self.colname].cat.categories)))
        else:
            path_info = None
        write_header = False
        rest_kwargs = self.kwargs.copy()
        if not is_first:
            if rest_kwargs.get('names', None) is None:
                write_header = True
            rest_kwargs.pop('skiprows', None)
            if rest_kwargs.get('header', 0) is not None:
                rest_kwargs.pop('header', None)
        if not is_last:
            rest_kwargs.pop('skipfooter', None)
        columns = self.full_columns
        project_after_read = False
        if self._columns is not None:
            if self.kwargs:
                project_after_read = True
            else:
                columns = self._columns
                rest_kwargs['usecols'] = columns
        df = pandas_read_text(self.reader, block, self.header, rest_kwargs, self.dtypes, columns, write_header, self.enforce, path_info)
        if project_after_read:
            return df[self.columns]
        return df