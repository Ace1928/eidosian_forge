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
def _execute_peer_access(kernel, args: Sequence['_array.DistributedArray'], kwargs: dict[str, '_array.DistributedArray']) -> '_array.DistributedArray':
    """Arguments must be in the replica mode."""
    assert len(args) >= 2
    if len(args) > 2:
        raise RuntimeError('Element-wise operation over more than two distributed arrays is not supported unless they share the same index_map.')
    if kwargs:
        raise RuntimeError('Keyword argument is not supported unless arguments share the same index_map.')
    args = list(args)
    for i, arg in enumerate(args):
        args[i] = arg._to_op_mode(_modes.REPLICA)
        for chunk in chain.from_iterable(args[i]._chunks_map.values()):
            chunk.flush(_modes.REPLICA)
    a, b = args
    if isinstance(kernel, cupy._core._kernel.ufunc):
        op = kernel._ops._guess_routine_from_in_types((a.dtype, b.dtype))
        if op is None:
            raise RuntimeError(f'Could not guess the return type of {kernel.name} with arguments of type {(a.dtype.type, b.dtype.type)}')
        out_types = op.out_types
    else:
        assert isinstance(kernel, cupy._core._kernel.ElementwiseKernel)
        _, out_types, _ = kernel._decide_params_type((a.dtype.type, b.dtype.type), ())
    if len(out_types) != 1:
        print(out_types)
        raise RuntimeError('Kernels returning other than signle array are not supported')
    dtype = out_types[0]
    shape = a.shape
    comms = a._comms
    out_chunks_map: dict[int, list[_chunk._Chunk]] = {}
    for a_chunk in chain.from_iterable(a._chunks_map.values()):
        a_dev = a_chunk.array.device.id
        with a_chunk.on_ready() as stream:
            out_array = _creation_basic.empty(a_chunk.array.shape, dtype)
            for b_chunk in chain.from_iterable(b._chunks_map.values()):
                intersection = _index_arith._index_intersection(a_chunk.index, b_chunk.index, shape)
                if intersection is None:
                    continue
                b_dev = b_chunk.array.device.id
                if cupy.cuda.runtime.deviceCanAccessPeer(a_dev, b_dev) != 1:
                    b_chunk = _array._make_chunk_async(b_dev, a_dev, b_chunk.index, b_chunk.array, b._comms)
                else:
                    cupy._core._kernel._check_peer_access(b_chunk.array, a_dev)
                stream.wait_event(b_chunk.ready)
                a_new_idx = _index_arith._index_for_subindex(a_chunk.index, intersection, shape)
                b_new_idx = _index_arith._index_for_subindex(b_chunk.index, intersection, shape)
                assert kernel.nin == 2
                kernel(typing.cast(ndarray, a_chunk.array)[a_new_idx], typing.cast(ndarray, b_chunk.array)[b_new_idx], out_array[a_new_idx])
            out_chunk = _chunk._Chunk(out_array, stream.record(), a_chunk.index, prevent_gc=b._chunks_map)
            out_chunks_map.setdefault(a_dev, []).append(out_chunk)
    return _array.DistributedArray(shape, dtype, out_chunks_map, _modes.REPLICA, comms)