import contextlib
import importlib
import os
import sys
import threading
import traceback
import warnings
from functools import lru_cache
from typing import Any, cast, List, Optional, Tuple, Union
import torch
import torch._C
from torch.types import Device
from .. import device as _device
from .._utils import classproperty
from ._utils import _dummy_type, _get_device_index
from .graphs import (
from .streams import Event, ExternalStream, Stream
from .memory import *  # noqa: F403
from .random import *  # noqa: F403
from torch.storage import _LegacyStorage, _warn_typed_storage_removal
from . import amp, jiterator, nvtx, profiler, sparse
class StreamContext:
    """Context-manager that selects a given stream.

    All CUDA kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """
    cur_stream: Optional['torch.cuda.Stream']

    def __init__(self, stream: Optional['torch.cuda.Stream']):
        self.stream = stream
        self.idx = _get_device_index(None, True)
        if not torch.jit.is_scripting():
            if self.idx is None:
                self.idx = -1
        self.src_prev_stream = None if not torch.jit.is_scripting() else torch.cuda.default_stream(None)
        self.dst_prev_stream = None if not torch.jit.is_scripting() else torch.cuda.default_stream(None)

    def __enter__(self):
        cur_stream = self.stream
        if cur_stream is None or self.idx == -1:
            return
        self.src_prev_stream = torch.cuda.current_stream(None)
        if self.src_prev_stream.device != cur_stream.device:
            with device(cur_stream.device):
                self.dst_prev_stream = torch.cuda.current_stream(cur_stream.device)
        torch.cuda.set_stream(cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        cur_stream = self.stream
        if cur_stream is None or self.idx == -1:
            return
        if self.src_prev_stream.device != cur_stream.device:
            torch.cuda.set_stream(self.dst_prev_stream)
        torch.cuda.set_stream(self.src_prev_stream)