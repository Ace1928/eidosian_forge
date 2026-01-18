import functools
from typing import Callable, Final, Optional
import numpy
import cupy
@functools.lru_cache
def _one_of(dtype):
    return dtype.type(1)