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
class CUDAPluggableAllocator(_CUDAAllocator):
    """CUDA memory allocator loaded from a so file."""

    def __init__(self, path_to_so_file: str, alloc_fn_name: str, free_fn_name: str):
        """Memory allocators are compiled in .so files and loaded dynamically using ctypes.

        To change the active allocator use the :func:`torch.memory.cuda.change_current_allocator` function.

        Args:
            path_to_so_file(str): Path in the filesystem to the `.so` file containing
                the allocator functions
            alloc_fn_name(str): Name of the function to perform the memory allocation
                in the so file. The signature must be:
                void* alloc_fn_name(ssize_t size, int device, cudaStream_t stream);
            free_fn_name(str): Name of the function to perform the memory release
                in the so file. The signature must be:
                void free_fn_name(void* ptr, size_t size, cudaStream_t stream);

        .. warning::
            This is currently supported only in unix OSs

        .. note::
            See :ref:`cuda-memory-management` for details on creating and using a custom allocator
        """
        allocator = ctypes.CDLL(path_to_so_file)
        alloc_fn = ctypes.cast(getattr(allocator, alloc_fn_name), ctypes.c_void_p).value
        free_fn = ctypes.cast(getattr(allocator, free_fn_name), ctypes.c_void_p).value
        assert alloc_fn is not None
        assert free_fn is not None
        self._allocator = torch._C._cuda_customAllocator(alloc_fn, free_fn)