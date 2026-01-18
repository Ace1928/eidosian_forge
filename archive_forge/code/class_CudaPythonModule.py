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
class CudaPythonModule(Module):

    def get_function(self, name):
        handle = driver.cuModuleGetFunction(self.handle, name.encode('utf8'))
        return CudaPythonFunction(weakref.proxy(self), handle, name)

    def get_global_symbol(self, name):
        ptr, size = driver.cuModuleGetGlobal(self.handle, name.encode('utf8'))
        return (MemoryPointer(self.context, ptr, size), size)