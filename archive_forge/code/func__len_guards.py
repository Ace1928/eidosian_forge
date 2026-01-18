from CuSignal under terms of the MIT license.
import warnings
from typing import Set
import cupy
import numpy as np
def _len_guards(M):
    """Handle small or incorrect window lengths"""
    if int(M) != M or M < 0:
        raise ValueError('Window length M must be a non-negative integer')
    return M <= 1