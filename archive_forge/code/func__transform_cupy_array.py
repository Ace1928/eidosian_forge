import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _transform_cupy_array(data: DataType) -> CupyT:
    import cupy
    if not hasattr(data, '__cuda_array_interface__') and hasattr(data, '__array__'):
        data = cupy.array(data, copy=False)
    if data.dtype.hasobject or data.dtype in [cupy.bool_]:
        data = data.astype(cupy.float32, copy=False)
    return data