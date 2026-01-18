import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _is_np_array_like(data: DataType) -> bool:
    return hasattr(data, '__array_interface__')