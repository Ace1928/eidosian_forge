import contextlib
import json
import os
import warnings
from io import BytesIO, IOBase, TextIOWrapper
from typing import Any, NamedTuple
import fsspec
import numpy as np
import pandas
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.concat import union_categoricals
from pandas.io.common import infer_compression
from pandas.util._decorators import doc
from modin.config import MinPartitionSize
from modin.core.io.file_dispatcher import OpenFile
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.db_conn import ModinDatabaseConnection
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.utils import ModinAssumptionError
@staticmethod
def _read_row_group_chunk(f, row_group_start, row_group_end, columns, filters, engine, to_pandas_kwargs):
    if engine == 'pyarrow':
        if filters is not None:
            import pyarrow.dataset as ds
            from pyarrow.parquet import filters_to_expression
            parquet_format = ds.ParquetFileFormat()
            fragment = parquet_format.make_fragment(f, row_groups=range(row_group_start, row_group_end))
            dataset = ds.FileSystemDataset([fragment], schema=fragment.physical_schema, format=parquet_format, filesystem=fragment.filesystem)
            metadata = dataset.schema.metadata or {}
            if b'pandas' in metadata and columns is not None:
                index_columns = json.loads(metadata[b'pandas'].decode('utf8'))['index_columns']
                index_columns = [col for col in index_columns if not isinstance(col, dict)]
                columns = list(columns) + list(set(index_columns) - set(columns))
            return dataset.to_table(columns=columns, filter=filters_to_expression(filters)).to_pandas(**to_pandas_kwargs)
        else:
            from pyarrow.parquet import ParquetFile
            return ParquetFile(f).read_row_groups(range(row_group_start, row_group_end), columns=columns, use_pandas_metadata=True).to_pandas(**to_pandas_kwargs)
    elif engine == 'fastparquet':
        from fastparquet import ParquetFile
        return ParquetFile(f)[row_group_start:row_group_end].to_pandas(columns=columns, filters=filters)
    else:
        raise ValueError(f"engine must be one of 'pyarrow', 'fastparquet', got: {engine}")