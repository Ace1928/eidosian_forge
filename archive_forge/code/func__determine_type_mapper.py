from __future__ import annotations
import json
import operator
import textwrap
import warnings
from collections import defaultdict
from datetime import datetime
from functools import reduce
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.fs as pa_fs
import pyarrow.parquet as pq
from fsspec.core import expand_paths_if_needed, stringify_path
from fsspec.implementations.arrow import ArrowFSWrapper
from pyarrow import dataset as pa_ds
from pyarrow import fs as pa_fs
import dask
from dask.base import normalize_token, tokenize
from dask.core import flatten
from dask.dataframe._compat import PANDAS_GE_220
from dask.dataframe.backends import pyarrow_schema_dispatch
from dask.dataframe.io.parquet.utils import (
from dask.dataframe.io.utils import _get_pyarrow_dtypes, _is_local_fs, _open_input_files
from dask.dataframe.utils import clear_known_categories, pyarrow_strings_enabled
from dask.delayed import Delayed
from dask.utils import getargspec, natural_sort_key
@classmethod
def _determine_type_mapper(cls, *, dtype_backend=None, convert_string=False, **kwargs):
    user_mapper = kwargs.get('arrow_to_pandas', {}).get('types_mapper')
    type_mappers = []

    def pyarrow_type_mapper(pyarrow_dtype):
        if PANDAS_GE_220 and pyarrow_dtype == pa.large_string():
            return pd.StringDtype('pyarrow')
        if pyarrow_dtype == pa.string():
            return pd.StringDtype('pyarrow')
        else:
            return pd.ArrowDtype(pyarrow_dtype)
    if user_mapper is not None:
        type_mappers.append(user_mapper)
    if convert_string:
        type_mappers.append({pa.string(): pd.StringDtype('pyarrow')}.get)
        if PANDAS_GE_220:
            type_mappers.append({pa.large_string(): pd.StringDtype('pyarrow')}.get)
        type_mappers.append({pa.date32(): pd.ArrowDtype(pa.date32())}.get)
        type_mappers.append({pa.date64(): pd.ArrowDtype(pa.date64())}.get)

        def _convert_decimal_type(type):
            if pa.types.is_decimal(type):
                return pd.ArrowDtype(type)
            return None
        type_mappers.append(_convert_decimal_type)
    if dtype_backend == 'numpy_nullable':
        type_mappers.append(PYARROW_NULLABLE_DTYPE_MAPPING.get)
    elif dtype_backend == 'pyarrow':
        type_mappers.append(pyarrow_type_mapper)

    def default_types_mapper(pyarrow_dtype):
        """Try all type mappers in order, starting from the user type mapper."""
        for type_converter in type_mappers:
            converted_type = type_converter(pyarrow_dtype)
            if converted_type is not None:
                return converted_type
    if len(type_mappers) > 0:
        return default_types_mapper