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
def _copy_chunks_map_in_replica_mode(self) -> dict[int, list[_Chunk]]:
    chunks_map = {}
    for dev, chunks in self._chunks_map.items():
        chunks_map[dev] = [chunk.copy() for chunk in chunks]
    if self._mode is not _modes.REPLICA:
        self._prepare_comms_and_streams(self._chunks_map.keys())
        _chunk._all_reduce_intersections(self._mode, self.shape, chunks_map, self._comms, self._streams)
    return chunks_map