import dataclasses
import typing
from typing import Callable, Optional
import cupy
from cupyx.distributed.array import _array
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _modes
def _pop_front_from_shape(arr) -> '_array.DistributedArray':
    assert arr.shape[0] == 1
    return _reshape_array_with(arr, lambda shape: shape[1:], lambda idx: idx[1:])