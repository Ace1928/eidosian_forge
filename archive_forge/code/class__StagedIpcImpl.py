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
class _StagedIpcImpl(object):
    """Implementation of GPU IPC using custom staging logic to workaround
    CUDA IPC limitation on peer accessibility between devices.
    """

    def __init__(self, parent, source_info):
        self.parent = parent
        self.base = parent.base
        self.handle = parent.handle
        self.size = parent.size
        self.source_info = source_info

    def open(self, context):
        from numba import cuda
        srcdev = Device.from_identity(self.source_info)
        if USE_NV_BINDING:
            srcdev_id = int(srcdev.id)
        else:
            srcdev_id = srcdev.id
        impl = _CudaIpcImpl(parent=self.parent)
        with cuda.gpus[srcdev_id]:
            source_ptr = impl.open(cuda.devices.get_context())
        newmem = context.memalloc(self.size)
        device_to_device(newmem, source_ptr, self.size)
        with cuda.gpus[srcdev_id]:
            impl.close()
        return newmem

    def close(self):
        pass