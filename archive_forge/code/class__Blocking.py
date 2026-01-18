import dataclasses
import typing
from typing import Callable, Optional
import cupy
from cupyx.distributed.array import _array
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _modes
@dataclasses.dataclass
class _Blocking:
    i_partitions: list[int]
    j_partitions: list[int]
    k_partitions: list[int]