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
def _check_valid_index_key(key: Union[int, slice, range, Iterable], size: int) -> None:
    if isinstance(key, int):
        if key < 0 and key + size < 0 or key >= size:
            raise IndexError(f'Invalid key: {key} is out of bounds for size {size}')
        return
    elif isinstance(key, slice):
        pass
    elif isinstance(key, range):
        if len(key) > 0:
            _check_valid_index_key(max(key), size=size)
            _check_valid_index_key(min(key), size=size)
    elif isinstance(key, Iterable):
        if len(key) > 0:
            _check_valid_index_key(int(max(key)), size=size)
            _check_valid_index_key(int(min(key)), size=size)
    else:
        _raise_bad_key_type(key)