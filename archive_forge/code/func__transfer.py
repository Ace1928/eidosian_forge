import contextlib
import dataclasses
from typing import Any, Iterable, Iterator
from cupy._core.core import ndarray
import cupy._creation.from_data as _creation_from_data
import cupy._creation.basic as _creation_basic
from cupy.cuda.device import Device
from cupy.cuda.stream import Event
from cupy.cuda.stream import Stream
from cupy.cuda.stream import get_current_stream
from cupy.cuda import nccl
from cupyx.distributed._nccl_comm import _get_nccl_dtype_and_count
def _transfer(src_comm: _Communicator, src_stream: Stream, src_data: _AsyncData, dst_comm: _Communicator, dst_stream: Stream, dst_dev: int) -> _AsyncData:
    src_dev = src_data.array.device.id
    if src_dev == dst_dev:
        return _AsyncData(src_data.array, src_data.ready)
    with Device(dst_dev):
        prev_stream = get_current_stream()
        try:
            dst_stream.use()
            dst_stream.wait_event(src_data.ready)
            dst_array = src_data.array.copy()
            return _AsyncData(dst_array, dst_stream.record(), prevent_gc=src_data.array)
        finally:
            prev_stream.use()