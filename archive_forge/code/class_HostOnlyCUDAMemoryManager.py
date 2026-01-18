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
class HostOnlyCUDAMemoryManager(BaseCUDAMemoryManager):
    """Base class for External Memory Management (EMM) Plugins that only
    implement on-device allocation. A subclass need not implement the
    ``memhostalloc`` and ``mempin`` methods.

    This class also implements ``reset`` and ``defer_cleanup`` (see
    :class:`numba.cuda.BaseCUDAMemoryManager`) for its own internal state
    management. If an EMM Plugin based on this class also implements these
    methods, then its implementations of these must also call the method from
    ``super()`` to give ``HostOnlyCUDAMemoryManager`` an opportunity to do the
    necessary work for the host allocations it is managing.

    This class does not implement ``interface_version``, as it will always be
    consistent with the version of Numba in which it is implemented. An EMM
    Plugin subclassing this class should implement ``interface_version``
    instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allocations = utils.UniqueDict()
        self.deallocations = _PendingDeallocs()

    def _attempt_allocation(self, allocator):
        """
        Attempt allocation by calling *allocator*.  If an out-of-memory error
        is raised, the pending deallocations are flushed and the allocation
        is retried.  If it fails in the second attempt, the error is reraised.
        """
        try:
            return allocator()
        except CudaAPIError as e:
            if USE_NV_BINDING:
                oom_code = binding.CUresult.CUDA_ERROR_OUT_OF_MEMORY
            else:
                oom_code = enums.CUDA_ERROR_OUT_OF_MEMORY
            if e.code == oom_code:
                self.deallocations.clear()
                return allocator()
            else:
                raise

    def memhostalloc(self, size, mapped=False, portable=False, wc=False):
        """Implements the allocation of pinned host memory.

        It is recommended that this method is not overridden by EMM Plugin
        implementations - instead, use the :class:`BaseCUDAMemoryManager`.
        """
        flags = 0
        if mapped:
            flags |= enums.CU_MEMHOSTALLOC_DEVICEMAP
        if portable:
            flags |= enums.CU_MEMHOSTALLOC_PORTABLE
        if wc:
            flags |= enums.CU_MEMHOSTALLOC_WRITECOMBINED
        if USE_NV_BINDING:

            def allocator():
                return driver.cuMemHostAlloc(size, flags)
            if mapped:
                pointer = self._attempt_allocation(allocator)
            else:
                pointer = allocator()
            alloc_key = pointer
        else:
            pointer = c_void_p()

            def allocator():
                driver.cuMemHostAlloc(byref(pointer), size, flags)
            if mapped:
                self._attempt_allocation(allocator)
            else:
                allocator()
            alloc_key = pointer.value
        finalizer = _hostalloc_finalizer(self, pointer, alloc_key, size, mapped)
        ctx = weakref.proxy(self.context)
        if mapped:
            mem = MappedMemory(ctx, pointer, size, finalizer=finalizer)
            self.allocations[alloc_key] = mem
            return mem.own()
        else:
            return PinnedMemory(ctx, pointer, size, finalizer=finalizer)

    def mempin(self, owner, pointer, size, mapped=False):
        """Implements the pinning of host memory.

        It is recommended that this method is not overridden by EMM Plugin
        implementations - instead, use the :class:`BaseCUDAMemoryManager`.
        """
        if isinstance(pointer, int) and (not USE_NV_BINDING):
            pointer = c_void_p(pointer)
        if USE_NV_BINDING:
            alloc_key = pointer
        else:
            alloc_key = pointer.value
        flags = 0
        if mapped:
            flags |= enums.CU_MEMHOSTREGISTER_DEVICEMAP

        def allocator():
            driver.cuMemHostRegister(pointer, size, flags)
        if mapped:
            self._attempt_allocation(allocator)
        else:
            allocator()
        finalizer = _pin_finalizer(self, pointer, alloc_key, mapped)
        ctx = weakref.proxy(self.context)
        if mapped:
            mem = MappedMemory(ctx, pointer, size, owner=owner, finalizer=finalizer)
            self.allocations[alloc_key] = mem
            return mem.own()
        else:
            return PinnedMemory(ctx, pointer, size, owner=owner, finalizer=finalizer)

    def memallocmanaged(self, size, attach_global):
        if USE_NV_BINDING:

            def allocator():
                ma_flags = binding.CUmemAttach_flags
                if attach_global:
                    flags = ma_flags.CU_MEM_ATTACH_GLOBAL.value
                else:
                    flags = ma_flags.CU_MEM_ATTACH_HOST.value
                return driver.cuMemAllocManaged(size, flags)
            ptr = self._attempt_allocation(allocator)
            alloc_key = ptr
        else:
            ptr = drvapi.cu_device_ptr()

            def allocator():
                flags = c_uint()
                if attach_global:
                    flags = enums.CU_MEM_ATTACH_GLOBAL
                else:
                    flags = enums.CU_MEM_ATTACH_HOST
                driver.cuMemAllocManaged(byref(ptr), size, flags)
            self._attempt_allocation(allocator)
            alloc_key = ptr.value
        finalizer = _alloc_finalizer(self, ptr, alloc_key, size)
        ctx = weakref.proxy(self.context)
        mem = ManagedMemory(ctx, ptr, size, finalizer=finalizer)
        self.allocations[alloc_key] = mem
        return mem.own()

    def reset(self):
        """Clears up all host memory (mapped and/or pinned) in the current
        context.

        EMM Plugins that override this method must call ``super().reset()`` to
        ensure that host allocations are also cleaned up."""
        self.allocations.clear()
        self.deallocations.clear()

    @contextlib.contextmanager
    def defer_cleanup(self):
        """Returns a context manager that disables cleanup of mapped or pinned
        host memory in the current context whilst it is active.

        EMM Plugins that override this method must obtain the context manager
        from this method before yielding to ensure that cleanup of host
        allocations is also deferred."""
        with self.deallocations.disable():
            yield