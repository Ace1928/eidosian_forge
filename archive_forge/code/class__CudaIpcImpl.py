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
class _CudaIpcImpl(object):
    """Implementation of GPU IPC using CUDA driver API.
    This requires the devices to be peer accessible.
    """

    def __init__(self, parent):
        self.base = parent.base
        self.handle = parent.handle
        self.size = parent.size
        self.offset = parent.offset
        self._opened_mem = None

    def open(self, context):
        """
        Import the IPC memory and returns a raw CUDA memory pointer object
        """
        if self.base is not None:
            raise ValueError('opening IpcHandle from original process')
        if self._opened_mem is not None:
            raise ValueError('IpcHandle is already opened')
        mem = context.open_ipc_handle(self.handle, self.offset + self.size)
        self._opened_mem = mem
        return mem.own().view(self.offset)

    def close(self):
        if self._opened_mem is None:
            raise ValueError('IpcHandle not opened')
        driver.cuIpcCloseMemHandle(self._opened_mem.handle)
        self._opened_mem = None