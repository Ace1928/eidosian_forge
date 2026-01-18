import functools
from typing import Callable, Final, Optional
import numpy
import cupy
@functools.lru_cache
def _zero_of(dtype):
    return dtype.type(0)