import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _array_interface(data: np.ndarray) -> bytes:
    assert data.dtype.hasobject is False, 'Input data contains `object` dtype.  Expecting numeric data.'
    interface = data.__array_interface__
    if 'mask' in interface:
        interface['mask'] = interface['mask'].__array_interface__
    interface_str = bytes(json.dumps(interface), 'utf-8')
    return interface_str