import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _meta_from_list(data: Sequence, field: str, dtype: Optional[NumpyDType], handle: ctypes.c_void_p) -> None:
    data_np = np.array(data)
    _meta_from_numpy(data_np, field, dtype, handle)