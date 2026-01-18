import typing
from typing import Sequence
from itertools import chain
import cupy
import cupy._creation.basic as _creation_basic
from cupy._core.core import ndarray
from cupy.cuda.device import Device
from cupy.cuda.stream import Stream
from cupy.cuda.stream import get_current_stream
from cupyx.distributed.array import _array
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _data_transfer
from cupyx.distributed.array import _index_arith
from cupyx.distributed.array import _modes
def _change_all_to_replica_mode(args: list['_array.DistributedArray'], kwargs: dict[str, '_array.DistributedArray']) -> None:
    args[:] = [arg._to_op_mode(_modes.REPLICA) for arg in args]
    kwargs.update(((k, arg._to_op_mode(_modes.REPLICA)) for k, arg in kwargs.items()))