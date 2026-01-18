from __future__ import annotations
import warnings
from typing import Callable
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.core.utils import is_duck_array, module_available
from xarray.namedarray import pycompat
from xarray.core.options import OPTIONS
def _is_contiguous(positions):
    """Given a non-empty list, does it consist of contiguous integers?"""
    previous = positions[0]
    for current in positions[1:]:
        if current != previous + 1:
            return False
        previous = current
    return True