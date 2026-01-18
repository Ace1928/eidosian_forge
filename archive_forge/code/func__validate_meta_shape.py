import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _validate_meta_shape(data: DataType, name: str) -> None:
    if hasattr(data, 'shape'):
        msg = f'Invalid shape: {data.shape} for {name}'
        if name in _matrix_meta:
            if len(data.shape) > 2:
                raise ValueError(msg)
            return
        if len(data.shape) > 2 or (len(data.shape) == 2 and (data.shape[1] != 0 and data.shape[1] != 1)):
            raise ValueError(f'Invalid shape: {data.shape} for {name}')