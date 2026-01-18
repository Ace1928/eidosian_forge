from __future__ import annotations
import re
from functools import total_ordering
from packaging.version import Version
import numpy as np
import pandas as pd
from numba import jit
from pandas.api.extensions import (
from numbers import Integral
from pandas.api.types import pandas_dtype, is_extension_array_dtype
@jit(nopython=True, nogil=True)
def _lexograph_lt(a1, a2):
    """
    Compare two 1D numpy arrays lexographically
    Parameters
    ----------
    a1: ndarray
        1D numpy array
    a2: ndarray
        1D numpy array

    Returns
    -------
    comparison:
        True if a1 < a2, False otherwise
    """
    for e1, e2 in zip(a1, a2):
        if e1 < e2:
            return True
        elif e1 > e2:
            return False
    return len(a1) < len(a2)