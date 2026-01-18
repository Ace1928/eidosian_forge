import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _is_dlpack(data: DataType) -> bool:
    return 'PyCapsule' in str(type(data)) and 'dltensor' in str(data)