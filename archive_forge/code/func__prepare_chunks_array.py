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
def _prepare_chunks_array(stream: Stream, args: Sequence['_array.DistributedArray'], kwargs: dict[str, '_array.DistributedArray'], dev: int, chunk_i: int) -> tuple[list[ndarray], dict[str, ndarray]]:

    def access_array(d_array):
        chunk = d_array._chunks_map[dev][chunk_i]
        stream.wait_event(chunk.ready)
        return chunk.array
    arg_arrays = [access_array(arg) for arg in args]
    kwarg_arrays = {key: access_array(arg) for key, arg in kwargs.items()}
    return (arg_arrays, kwarg_arrays)