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
def _extract_index_level(table, result_table, field_name, field_name_to_metadata, types_mapper=None):
    logical_name = field_name_to_metadata[field_name]['name']
    index_name = _backwards_compatible_index_name(field_name, logical_name)
    i = table.schema.get_field_index(field_name)
    if i == -1:
        return (result_table, None, None)
    pd = _pandas_api.pd
    col = table.column(i)
    values = col.to_pandas(types_mapper=types_mapper).values
    if hasattr(values, 'flags') and (not values.flags.writeable):
        values = values.copy()
    if isinstance(col.type, pa.lib.TimestampType) and col.type.tz is not None:
        index_level = make_tz_aware(pd.Series(values, copy=False), col.type.tz)
    else:
        index_level = pd.Series(values, dtype=values.dtype, copy=False)
    result_table = result_table.remove_column(result_table.schema.get_field_index(field_name))
    return (result_table, index_level, index_name)