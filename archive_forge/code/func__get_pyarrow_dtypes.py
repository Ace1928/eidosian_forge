from __future__ import annotations
import json
from typing import Protocol, runtime_checkable
from uuid import uuid4
import fsspec
import pandas as pd
from fsspec.implementations.local import LocalFileSystem
from packaging.version import parse as parse_version
def _get_pyarrow_dtypes(schema, categories, dtype_backend=None):
    """Convert a pyarrow.Schema object to pandas dtype dict"""
    if dtype_backend == 'numpy_nullable':
        from dask.dataframe.io.parquet.arrow import PYARROW_NULLABLE_DTYPE_MAPPING
        type_mapper = PYARROW_NULLABLE_DTYPE_MAPPING.get
    else:
        type_mapper = lambda t: t.to_pandas_dtype()
    has_pandas_metadata = schema.metadata is not None and b'pandas' in schema.metadata
    if has_pandas_metadata:
        pandas_metadata = json.loads(schema.metadata[b'pandas'].decode('utf8'))
        pandas_metadata_dtypes = {c.get('field_name', c.get('name', None)): c['numpy_type'] for c in pandas_metadata.get('columns', [])}
        tz = {c.get('field_name', c.get('name', None)): c['metadata'].get('timezone', None) for c in pandas_metadata.get('columns', []) if c['pandas_type'] in ('datetime', 'datetimetz') and c['metadata']}
    else:
        pandas_metadata_dtypes = {}
    dtypes = {}
    for i in range(len(schema)):
        field = schema[i]
        if field.name in pandas_metadata_dtypes:
            if field.name in tz:
                numpy_dtype = pd.Series([], dtype='M8[ns]').dt.tz_localize(tz[field.name]).dtype
            else:
                numpy_dtype = pandas_metadata_dtypes[field.name]
        else:
            try:
                numpy_dtype = type_mapper(field.type)
            except NotImplementedError:
                continue
        dtypes[field.name] = numpy_dtype
    if categories:
        for cat in categories:
            dtypes[cat] = 'category'
    return dtypes