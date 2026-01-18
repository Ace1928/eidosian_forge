import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _is_cudf_ser(data: DataType) -> bool:
    return lazy_isinstance(data, 'cudf.core.series', 'Series')