from itertools import chain
from typing import Any, Callable, Iterable, Optional
import numpy
from numpy.typing import ArrayLike
from numpy.typing import DTypeLike
import cupy
from cupy._core.core import ndarray
import cupy._creation.from_data as _creation_from_data
import cupy._core._routines_math as _math
import cupy._core._routines_statistics as _statistics
from cupy.cuda.device import Device
from cupy.cuda.stream import Stream
from cupy.cuda.stream import get_current_stream
from cupyx.distributed.array import _chunk
from cupyx.distributed.array._chunk import _Chunk
from cupyx.distributed.array import _data_transfer
from cupyx.distributed.array._data_transfer import _Communicator
from cupyx.distributed.array import _elementwise
from cupyx.distributed.array import _index_arith
from cupyx.distributed.array import _modes
from cupyx.distributed.array import _reduction
from cupyx.distributed.array import _linalg
def _copy_chunks_map_in_op_mode(self, op_mode: _modes._OpMode) -> dict[int, list[_Chunk]]:
    chunks_map = self._copy_chunks_map_in_replica_mode()
    for chunk in chain.from_iterable(chunks_map.values()):
        chunk.flush(_modes.REPLICA)
    chunks_list = list(chain.from_iterable(chunks_map.values()))
    identity = op_mode.identity_of(self.dtype)
    for i in range(len(chunks_list)):
        a_chunk = chunks_list[i]
        for j in range(i + 1, len(chunks_list)):
            b_chunk = chunks_list[j]
            a_chunk.set_identity_on_intersection(b_chunk.index, self.shape, identity)
    return chunks_map