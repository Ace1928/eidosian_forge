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
class PyArrowImpl(BaseImpl):

    def __init__(self) -> None:
        import_optional_dependency('pyarrow', extra='pyarrow is required for parquet support.')
        import pyarrow.parquet
        import pandas.core.arrays.arrow.extension_types
        self.api = pyarrow

    def write(self, df: DataFrame, path: FilePath | WriteBuffer[bytes], compression: str | None='snappy', index: bool | None=None, storage_options: StorageOptions | None=None, partition_cols: list[str] | None=None, filesystem=None, **kwargs) -> None:
        self.validate_dataframe(df)
        from_pandas_kwargs: dict[str, Any] = {'schema': kwargs.pop('schema', None)}
        if index is not None:
            from_pandas_kwargs['preserve_index'] = index
        table = self.api.Table.from_pandas(df, **from_pandas_kwargs)
        if df.attrs:
            df_metadata = {'PANDAS_ATTRS': json.dumps(df.attrs)}
            existing_metadata = table.schema.metadata
            merged_metadata = {**existing_metadata, **df_metadata}
            table = table.replace_schema_metadata(merged_metadata)
        path_or_handle, handles, filesystem = _get_path_or_handle(path, filesystem, storage_options=storage_options, mode='wb', is_dir=partition_cols is not None)
        if isinstance(path_or_handle, io.BufferedWriter) and hasattr(path_or_handle, 'name') and isinstance(path_or_handle.name, (str, bytes)):
            if isinstance(path_or_handle.name, bytes):
                path_or_handle = path_or_handle.name.decode()
            else:
                path_or_handle = path_or_handle.name
        try:
            if partition_cols is not None:
                self.api.parquet.write_to_dataset(table, path_or_handle, compression=compression, partition_cols=partition_cols, filesystem=filesystem, **kwargs)
            else:
                self.api.parquet.write_table(table, path_or_handle, compression=compression, filesystem=filesystem, **kwargs)
        finally:
            if handles is not None:
                handles.close()

    def read(self, path, columns=None, filters=None, use_nullable_dtypes: bool=False, dtype_backend: DtypeBackend | lib.NoDefault=lib.no_default, storage_options: StorageOptions | None=None, filesystem=None, **kwargs) -> DataFrame:
        kwargs['use_pandas_metadata'] = True
        to_pandas_kwargs = {}
        if dtype_backend == 'numpy_nullable':
            from pandas.io._util import _arrow_dtype_mapping
            mapping = _arrow_dtype_mapping()
            to_pandas_kwargs['types_mapper'] = mapping.get
        elif dtype_backend == 'pyarrow':
            to_pandas_kwargs['types_mapper'] = pd.ArrowDtype
        elif using_pyarrow_string_dtype():
            to_pandas_kwargs['types_mapper'] = arrow_string_types_mapper()
        manager = _get_option('mode.data_manager', silent=True)
        if manager == 'array':
            to_pandas_kwargs['split_blocks'] = True
        path_or_handle, handles, filesystem = _get_path_or_handle(path, filesystem, storage_options=storage_options, mode='rb')
        try:
            pa_table = self.api.parquet.read_table(path_or_handle, columns=columns, filesystem=filesystem, filters=filters, **kwargs)
            result = pa_table.to_pandas(**to_pandas_kwargs)
            if manager == 'array':
                result = result._as_manager('array', copy=False)
            if pa_table.schema.metadata:
                if b'PANDAS_ATTRS' in pa_table.schema.metadata:
                    df_metadata = pa_table.schema.metadata[b'PANDAS_ATTRS']
                    result.attrs = json.loads(df_metadata)
            return result
        finally:
            if handles is not None:
                handles.close()