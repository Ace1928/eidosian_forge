import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _warn_unused_missing(data: DataType, missing: Optional[FloatCompatible]) -> None:
    if missing is not None and (not np.isnan(missing)):
        warnings.warn('`missing` is not used for current input data type:' + str(type(data)), UserWarning)