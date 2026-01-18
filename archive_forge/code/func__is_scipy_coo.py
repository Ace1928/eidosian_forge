import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _is_scipy_coo(data: DataType) -> bool:
    try:
        import scipy.sparse
    except ImportError:
        return False
    return isinstance(data, scipy.sparse.coo_matrix)