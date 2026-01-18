from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _arrow_table_reduce(t: 'pyarrow.Table'):
    """Custom reducer for Arrow Tables that works around a zero-copy slice pickling bug.
    Background:
        Arrow has both array-level slicing and buffer-level slicing; both are zero-copy,
        but the former has a serialization bug where the entire buffer is serialized
        instead of just the slice, while the latter's serialization works as expected
        and only serializes the slice of the buffer. I.e., array-level slicing doesn't
        propagate the slice down to the buffer when serializing the array.
        All that these copy methods do is, at serialization time, take the array-level
        slicing and translate them to buffer-level slicing, so only the buffer slice is
        sent over the wire instead of the entire buffer.
    See https://issues.apache.org/jira/browse/ARROW-10739.
    """
    global _serialization_fallback_set
    reduced_columns = []
    for column_name in t.column_names:
        column = t[column_name]
        try:
            reduced_column = _arrow_chunked_array_reduce(column)
        except Exception as e:
            if not _is_dense_union(column.type) and _is_in_test():
                raise e from None
            if type(column.type) not in _serialization_fallback_set:
                logger.warning(f"Failed to complete optimized serialization of Arrow Table, serialization of column '{column_name}' of type {column.type} failed, so we're falling back to Arrow IPC serialization for the table. Note that this may result in slower serialization and more worker memory utilization. Serialization error:", exc_info=True)
                _serialization_fallback_set.add(type(column.type))
            return _arrow_table_ipc_reduce(t)
        else:
            reduced_columns.append(reduced_column)
    return (_reconstruct_table, (reduced_columns, t.schema))