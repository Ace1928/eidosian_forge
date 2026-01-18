import sys
import os
import ctypes
import weakref
import functools
import warnings
import logging
import threading
import asyncio
import pathlib
from itertools import product
from abc import ABCMeta, abstractmethod
from ctypes import (c_int, byref, c_size_t, c_char, c_char_p, addressof,
import contextlib
import importlib
import numpy as np
from collections import namedtuple, deque
from numba import mviewbuf
from numba.core import utils, serialize, config
from .error import CudaSupportError, CudaDriverError
from .drvapi import API_PROTOTYPES
from .drvapi import cu_occupancy_b2d_size, cu_stream_callback_pyobj, cu_uuid
from numba.cuda.cudadrv import enums, drvapi, nvrtc, _extras
def _pin_finalizer(memory_manager, ptr, alloc_key, mapped):
    """
    Finalize temporary page-locking of host memory by `context.mempin`.

    This applies to memory not otherwise managed by CUDA. Page-locking can
    be requested multiple times on the same memory, and must therefore be
    lifted as soon as finalization is requested, otherwise subsequent calls to
    `mempin` may fail with `CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED`, leading
    to unexpected behavior for the context managers `cuda.{pinned,mapped}`.
    This function therefore carries out finalization immediately, bypassing the
    `context.deallocations` queue.

    """
    allocations = memory_manager.allocations

    def core():
        if mapped and allocations:
            del allocations[alloc_key]
        driver.cuMemHostUnregister(ptr)
    return core