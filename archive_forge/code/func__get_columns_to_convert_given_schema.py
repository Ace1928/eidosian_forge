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
def _get_columns_to_convert_given_schema(df, schema, preserve_index):
    """
    Specialized version of _get_columns_to_convert in case a Schema is
    specified.
    In that case, the Schema is used as the single point of truth for the
    table structure (types, which columns are included, order of columns, ...).
    """
    column_names = []
    columns_to_convert = []
    convert_fields = []
    index_descriptors = []
    index_column_names = []
    index_levels = []
    for name in schema.names:
        try:
            col = df[name]
            is_index = False
        except KeyError:
            try:
                col = _get_index_level(df, name)
            except (KeyError, IndexError):
                raise KeyError("name '{}' present in the specified schema is not found in the columns or index".format(name))
            if preserve_index is False:
                raise ValueError("name '{}' present in the specified schema corresponds to the index, but 'preserve_index=False' was specified".format(name))
            elif preserve_index is None and isinstance(col, _pandas_api.pd.RangeIndex):
                raise ValueError("name '{}' is present in the schema, but it is a RangeIndex which will not be converted as a column in the Table, but saved as metadata-only not in columns. Specify 'preserve_index=True' to force it being added as a column, or remove it from the specified schema".format(name))
            is_index = True
        name = _column_name_to_strings(name)
        if _pandas_api.is_sparse(col):
            raise TypeError('Sparse pandas data (column {}) not supported.'.format(name))
        field = schema.field(name)
        columns_to_convert.append(col)
        convert_fields.append(field)
        column_names.append(name)
        if is_index:
            index_column_names.append(name)
            index_descriptors.append(name)
            index_levels.append(col)
    all_names = column_names + index_column_names
    return (all_names, column_names, index_column_names, index_descriptors, index_levels, columns_to_convert, convert_fields)