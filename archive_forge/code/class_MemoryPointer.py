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
class MemoryPointer(object):
    """A memory pointer that owns a buffer, with an optional finalizer. Memory
    pointers provide reference counting, and instances are initialized with a
    reference count of 1.

    The base ``MemoryPointer`` class does not use the
    reference count for managing the buffer lifetime. Instead, the buffer
    lifetime is tied to the memory pointer instance's lifetime:

    - When the instance is deleted, the finalizer will be called.
    - When the reference count drops to 0, no action is taken.

    Subclasses of ``MemoryPointer`` may modify these semantics, for example to
    tie the buffer lifetime to the reference count, so that the buffer is freed
    when there are no more references.

    :param context: The context in which the pointer was allocated.
    :type context: Context
    :param pointer: The address of the buffer.
    :type pointer: ctypes.c_void_p
    :param size: The size of the allocation in bytes.
    :type size: int
    :param owner: The owner is sometimes set by the internals of this class, or
                  used for Numba's internal memory management. It should not be
                  provided by an external user of the ``MemoryPointer`` class
                  (e.g. from within an EMM Plugin); the default of `None`
                  should always suffice.
    :type owner: NoneType
    :param finalizer: A function that is called when the buffer is to be freed.
    :type finalizer: function
    """
    __cuda_memory__ = True

    def __init__(self, context, pointer, size, owner=None, finalizer=None):
        self.context = context
        self.device_pointer = pointer
        self.size = size
        self._cuda_memsize_ = size
        self.is_managed = finalizer is not None
        self.refct = 1
        self.handle = self.device_pointer
        self._owner = owner
        if finalizer is not None:
            self._finalizer = weakref.finalize(self, finalizer)

    @property
    def owner(self):
        return self if self._owner is None else self._owner

    def own(self):
        return OwnedPointer(weakref.proxy(self))

    def free(self):
        """
        Forces the device memory to the trash.
        """
        if self.is_managed:
            if not self._finalizer.alive:
                raise RuntimeError('Freeing dead memory')
            self._finalizer()
            assert not self._finalizer.alive

    def memset(self, byte, count=None, stream=0):
        count = self.size if count is None else count
        if stream:
            driver.cuMemsetD8Async(self.device_pointer, byte, count, stream.handle)
        else:
            driver.cuMemsetD8(self.device_pointer, byte, count)

    def view(self, start, stop=None):
        if stop is None:
            size = self.size - start
        else:
            size = stop - start
        if not self.device_pointer_value:
            if size != 0:
                raise RuntimeError('non-empty slice into empty slice')
            view = self
        else:
            base = self.device_pointer_value + start
            if size < 0:
                raise RuntimeError('size cannot be negative')
            if USE_NV_BINDING:
                pointer = binding.CUdeviceptr()
                ctypes_ptr = drvapi.cu_device_ptr.from_address(pointer.getPtr())
                ctypes_ptr.value = base
            else:
                pointer = drvapi.cu_device_ptr(base)
            view = MemoryPointer(self.context, pointer, size, owner=self.owner)
        if isinstance(self.owner, (MemoryPointer, OwnedPointer)):
            return OwnedPointer(weakref.proxy(self.owner), view)
        else:
            return view

    @property
    def device_ctypes_pointer(self):
        return self.device_pointer

    @property
    def device_pointer_value(self):
        if USE_NV_BINDING:
            return int(self.device_pointer) or None
        else:
            return self.device_pointer.value