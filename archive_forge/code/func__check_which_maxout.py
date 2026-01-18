import operator
import re
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy
from ..compat import cupy, has_cupy_gpu
def _check_which_maxout(which, B: int, I: int, P: int):
    shape = (B, I)
    msg = 'maximum index (which) should be encoded as 32-bit integers'
    assert which.dtype == 'int32', msg
    if which.shape != shape:
        msg = f'maximum index (which) has incorrect shape, expected: {shape}, was: {which.shape}'
        raise ValueError(msg)
    if not _values_within_range(which, 0, P):
        raise IndexError('maximum index (which) value out of bounds')