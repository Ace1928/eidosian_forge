from __future__ import annotations
import io
import json
import os
from typing import (
import warnings
from warnings import catch_warnings
from pandas._config import using_pyarrow_string_dtype
from pandas._config.config import _get_option
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import AbstractMethodError
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
import pandas as pd
from pandas import (
from pandas.core.shared_docs import _shared_docs
from pandas.io._util import arrow_string_types_mapper
from pandas.io.common import (
class FastParquetImpl(BaseImpl):

    def __init__(self) -> None:
        fastparquet = import_optional_dependency('fastparquet', extra='fastparquet is required for parquet support.')
        self.api = fastparquet

    def write(self, df: DataFrame, path, compression: Literal['snappy', 'gzip', 'brotli'] | None='snappy', index=None, partition_cols=None, storage_options: StorageOptions | None=None, filesystem=None, **kwargs) -> None:
        self.validate_dataframe(df)
        if 'partition_on' in kwargs and partition_cols is not None:
            raise ValueError('Cannot use both partition_on and partition_cols. Use partition_cols for partitioning data')
        if 'partition_on' in kwargs:
            partition_cols = kwargs.pop('partition_on')
        if partition_cols is not None:
            kwargs['file_scheme'] = 'hive'
        if filesystem is not None:
            raise NotImplementedError('filesystem is not implemented for the fastparquet engine.')
        path = stringify_path(path)
        if is_fsspec_url(path):
            fsspec = import_optional_dependency('fsspec')
            kwargs['open_with'] = lambda path, _: fsspec.open(path, 'wb', **storage_options or {}).open()
        elif storage_options:
            raise ValueError('storage_options passed with file object or non-fsspec file path')
        with catch_warnings(record=True):
            self.api.write(path, df, compression=compression, write_index=index, partition_on=partition_cols, **kwargs)

    def read(self, path, columns=None, filters=None, storage_options: StorageOptions | None=None, filesystem=None, **kwargs) -> DataFrame:
        parquet_kwargs: dict[str, Any] = {}
        use_nullable_dtypes = kwargs.pop('use_nullable_dtypes', False)
        dtype_backend = kwargs.pop('dtype_backend', lib.no_default)
        parquet_kwargs['pandas_nulls'] = False
        if use_nullable_dtypes:
            raise ValueError("The 'use_nullable_dtypes' argument is not supported for the fastparquet engine")
        if dtype_backend is not lib.no_default:
            raise ValueError("The 'dtype_backend' argument is not supported for the fastparquet engine")
        if filesystem is not None:
            raise NotImplementedError('filesystem is not implemented for the fastparquet engine.')
        path = stringify_path(path)
        handles = None
        if is_fsspec_url(path):
            fsspec = import_optional_dependency('fsspec')
            parquet_kwargs['fs'] = fsspec.open(path, 'rb', **storage_options or {}).fs
        elif isinstance(path, str) and (not os.path.isdir(path)):
            handles = get_handle(path, 'rb', is_text=False, storage_options=storage_options)
            path = handles.handle
        try:
            parquet_file = self.api.ParquetFile(path, **parquet_kwargs)
            return parquet_file.to_pandas(columns=columns, filters=filters, **kwargs)
        finally:
            if handles is not None:
                handles.close()