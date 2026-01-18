import collections
import contextlib
import ctypes
import pickle
import sys
import warnings
from inspect import signature
from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import _C
from torch.types import Device
from . import _get_device_index, _get_nvml_device_index, _lazy_init, is_initialized
from ._memory_viz import memory as _memory, segments as _segments
from ._utils import _dummy_type
def _get_current_allocator() -> _CUDAAllocator:
    """Return the allocator being currently used.

    .. note::
        See :ref:`cuda-memory-management` for details on creating and using a custom allocator
    """
    return _CUDAAllocator(torch._C._cuda_getAllocator())