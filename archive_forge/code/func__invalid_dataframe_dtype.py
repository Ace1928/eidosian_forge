import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _invalid_dataframe_dtype(data: DataType) -> None:
    if hasattr(data, 'dtypes') and hasattr(data.dtypes, '__iter__'):
        bad_fields = [f'{data.columns[i]}: {dtype}' for i, dtype in enumerate(data.dtypes) if dtype.name not in _pandas_dtype_mapper]
        err = ' Invalid columns:' + ', '.join(bad_fields)
    else:
        err = ''
    type_err = 'DataFrame.dtypes for data must be int, float, bool or category.'
    msg = f'{type_err} {_ENABLE_CAT_ERR} {err}'
    raise ValueError(msg)