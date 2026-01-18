import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _meta_from_dt(data: DataType, field: str, dtype: Optional[NumpyDType], handle: ctypes.c_void_p) -> None:
    data, _, _ = _transform_dt_df(data, None, None, field, dtype)
    _meta_from_numpy(data, field, dtype, handle)