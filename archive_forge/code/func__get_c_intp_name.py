from __future__ import annotations
from collections.abc import Iterable
from typing import Final, TYPE_CHECKING, Callable
import numpy as np
def _get_c_intp_name() -> str:
    char = np.dtype('p').char
    if char == 'i':
        return 'c_int'
    elif char == 'l':
        return 'c_long'
    elif char == 'q':
        return 'c_longlong'
    else:
        return 'c_long'