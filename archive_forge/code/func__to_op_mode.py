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
def _to_op_mode(self, op_mode: _modes.Mode) -> 'DistributedArray':
    if self._mode is op_mode:
        return self
    if len(self._chunks_map) == 1:
        chunks, = self._chunks_map.values()
        if len(chunks) == 1:
            chunks[0].flush(self._mode)
            return DistributedArray(self.shape, self.dtype, self._chunks_map, op_mode, self._comms)
    if op_mode is _modes.REPLICA:
        chunks_map = self._copy_chunks_map_in_replica_mode()
    else:
        assert op_mode is not None
        chunks_map = self._copy_chunks_map_in_op_mode(op_mode)
    return DistributedArray(self.shape, self.dtype, chunks_map, op_mode, self._comms)