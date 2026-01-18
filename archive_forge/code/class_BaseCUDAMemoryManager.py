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
class BaseCUDAMemoryManager(object, metaclass=ABCMeta):
    """Abstract base class for External Memory Management (EMM) Plugins."""

    def __init__(self, *args, **kwargs):
        if 'context' not in kwargs:
            raise RuntimeError('Memory manager requires a context')
        self.context = kwargs.pop('context')

    @abstractmethod
    def memalloc(self, size):
        """
        Allocate on-device memory in the current context.

        :param size: Size of allocation in bytes
        :type size: int
        :return: A memory pointer instance that owns the allocated memory
        :rtype: :class:`MemoryPointer`
        """

    @abstractmethod
    def memhostalloc(self, size, mapped, portable, wc):
        """
        Allocate pinned host memory.

        :param size: Size of the allocation in bytes
        :type size: int
        :param mapped: Whether the allocated memory should be mapped into the
                       CUDA address space.
        :type mapped: bool
        :param portable: Whether the memory will be considered pinned by all
                         contexts, and not just the calling context.
        :type portable: bool
        :param wc: Whether to allocate the memory as write-combined.
        :type wc: bool
        :return: A memory pointer instance that owns the allocated memory. The
                 return type depends on whether the region was mapped into
                 device memory.
        :rtype: :class:`MappedMemory` or :class:`PinnedMemory`
        """

    @abstractmethod
    def mempin(self, owner, pointer, size, mapped):
        """
        Pin a region of host memory that is already allocated.

        :param owner: The object that owns the memory.
        :param pointer: The pointer to the beginning of the region to pin.
        :type pointer: int
        :param size: The size of the region in bytes.
        :type size: int
        :param mapped: Whether the region should also be mapped into device
                       memory.
        :type mapped: bool
        :return: A memory pointer instance that refers to the allocated
                 memory.
        :rtype: :class:`MappedMemory` or :class:`PinnedMemory`
        """

    @abstractmethod
    def initialize(self):
        """
        Perform any initialization required for the EMM plugin instance to be
        ready to use.

        :return: None
        """

    @abstractmethod
    def get_ipc_handle(self, memory):
        """
        Return an IPC handle from a GPU allocation.

        :param memory: Memory for which the IPC handle should be created.
        :type memory: :class:`MemoryPointer`
        :return: IPC handle for the allocation
        :rtype: :class:`IpcHandle`
        """

    @abstractmethod
    def get_memory_info(self):
        """
        Returns ``(free, total)`` memory in bytes in the context. May raise
        :class:`NotImplementedError`, if returning such information is not
        practical (e.g. for a pool allocator).

        :return: Memory info
        :rtype: :class:`MemoryInfo`
        """

    @abstractmethod
    def reset(self):
        """
        Clears up all memory allocated in this context.

        :return: None
        """

    @abstractmethod
    def defer_cleanup(self):
        """
        Returns a context manager that ensures the implementation of deferred
        cleanup whilst it is active.

        :return: Context manager
        """

    @property
    @abstractmethod
    def interface_version(self):
        """
        Returns an integer specifying the version of the EMM Plugin interface
        supported by the plugin implementation. Should always return 1 for
        implementations of this version of the specification.
        """