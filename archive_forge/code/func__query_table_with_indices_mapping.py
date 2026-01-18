from collections.abc import Mapping, MutableMapping
from functools import partial
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, TypeVar, Union
import numpy as np
import pandas as pd
import pyarrow as pa
from packaging import version
from .. import config
from ..features import Features
from ..features.features import _ArrayXDExtensionType, _is_zero_copy_only, decode_nested_example, pandas_types_mapper
from ..table import Table
from ..utils.py_utils import no_op_if_value_is_null
def _query_table_with_indices_mapping(table: Table, key: Union[int, slice, range, str, Iterable], indices: Table) -> pa.Table:
    """
    Query a pyarrow Table to extract the subtable that correspond to the given key.
    The :obj:`indices` parameter corresponds to the indices mapping in case we cant to take into
    account a shuffling or an indices selection for example.
    The indices table must contain one column named "indices" of type uint64.
    """
    if isinstance(key, int):
        key = indices.fast_slice(key % indices.num_rows, 1).column(0)[0].as_py()
        return _query_table(table, key)
    if isinstance(key, slice):
        key = range(*key.indices(indices.num_rows))
    if isinstance(key, range):
        if _is_range_contiguous(key) and key.start >= 0:
            return _query_table(table, [i.as_py() for i in indices.fast_slice(key.start, key.stop - key.start).column(0)])
        else:
            pass
    if isinstance(key, str):
        table = table.select([key])
        return _query_table(table, indices.column(0).to_pylist())
    if isinstance(key, Iterable):
        return _query_table(table, [indices.fast_slice(i, 1).column(0)[0].as_py() for i in key])
    _raise_bad_key_type(key)