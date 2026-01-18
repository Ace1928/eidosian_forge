import ast
from collections.abc import Sequence
from concurrent import futures
import concurrent.futures.thread  # noqa
from copy import deepcopy
from itertools import zip_longest
import json
import operator
import re
import warnings
import numpy as np
import pyarrow as pa
from pyarrow.lib import _pandas_api, frombytes  # noqa
def _reconstruct_columns_from_metadata(columns, column_indexes):
    """Construct a pandas MultiIndex from `columns` and column index metadata
    in `column_indexes`.

    Parameters
    ----------
    columns : List[pd.Index]
        The columns coming from a pyarrow.Table
    column_indexes : List[Dict[str, str]]
        The column index metadata deserialized from the JSON schema metadata
        in a :class:`~pyarrow.Table`.

    Returns
    -------
    result : MultiIndex
        The index reconstructed using `column_indexes` metadata with levels of
        the correct type.

    Notes
    -----
    * Part of :func:`~pyarrow.pandas_compat.table_to_blockmanager`
    """
    pd = _pandas_api.pd
    levels = getattr(columns, 'levels', None) or [columns]
    labels = _get_multiindex_codes(columns) or [pd.RangeIndex(len(level)) for level in levels]
    levels_dtypes = [(level, col_index.get('pandas_type', str(level.dtype)), col_index.get('numpy_type', None)) for level, col_index in zip_longest(levels, column_indexes, fillvalue={})]
    new_levels = []
    encoder = operator.methodcaller('encode', 'UTF-8')
    for level, pandas_dtype, numpy_dtype in levels_dtypes:
        dtype = _pandas_type_to_numpy_type(pandas_dtype)
        if dtype == np.bytes_:
            level = level.map(encoder)
        if pandas_dtype == 'datetimetz':
            tz = pa.lib.string_to_tzinfo(column_indexes[0]['metadata']['timezone'])
            level = pd.to_datetime(level, utc=True).tz_convert(tz)
        elif level.dtype != dtype:
            level = level.astype(dtype)
        if level.dtype != numpy_dtype and pandas_dtype != 'datetimetz':
            level = level.astype(numpy_dtype)
        new_levels.append(level)
    return pd.MultiIndex(new_levels, labels, names=columns.names)