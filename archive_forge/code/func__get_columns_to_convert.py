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
def _get_columns_to_convert(df, schema, preserve_index, columns):
    columns = _resolve_columns_of_interest(df, schema, columns)
    if not df.columns.is_unique:
        raise ValueError('Duplicate column names found: {}'.format(list(df.columns)))
    if schema is not None:
        return _get_columns_to_convert_given_schema(df, schema, preserve_index)
    column_names = []
    index_levels = _get_index_level_values(df.index) if preserve_index is not False else []
    columns_to_convert = []
    convert_fields = []
    for name in columns:
        col = df[name]
        name = _column_name_to_strings(name)
        if _pandas_api.is_sparse(col):
            raise TypeError('Sparse pandas data (column {}) not supported.'.format(name))
        columns_to_convert.append(col)
        convert_fields.append(None)
        column_names.append(name)
    index_descriptors = []
    index_column_names = []
    for i, index_level in enumerate(index_levels):
        name = _index_level_name(index_level, i, column_names)
        if isinstance(index_level, _pandas_api.pd.RangeIndex) and preserve_index is None:
            descr = _get_range_index_descriptor(index_level)
        else:
            columns_to_convert.append(index_level)
            convert_fields.append(None)
            descr = name
            index_column_names.append(name)
        index_descriptors.append(descr)
    all_names = column_names + index_column_names
    return (all_names, column_names, index_column_names, index_descriptors, index_levels, columns_to_convert, convert_fields)