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
def _is_peer_access_needed(args: Sequence['_array.DistributedArray'], kwargs: dict[str, '_array.DistributedArray']) -> bool:
    index_map = None
    for arg in chain(args, kwargs.values()):
        if index_map is None:
            index_map = arg.index_map
        elif arg.index_map != index_map:
            return True
    return False