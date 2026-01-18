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
class Linker(metaclass=ABCMeta):
    """Abstract base class for linkers"""

    @classmethod
    def new(cls, max_registers=0, lineinfo=False, cc=None):
        if config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY:
            return MVCLinker(max_registers, lineinfo, cc)
        elif USE_NV_BINDING:
            return CudaPythonLinker(max_registers, lineinfo, cc)
        else:
            return CtypesLinker(max_registers, lineinfo, cc)

    @abstractmethod
    def __init__(self, max_registers, lineinfo, cc):
        pass

    @property
    @abstractmethod
    def info_log(self):
        """Return the info log from the linker invocation"""

    @property
    @abstractmethod
    def error_log(self):
        """Return the error log from the linker invocation"""

    @abstractmethod
    def add_ptx(self, ptx, name):
        """Add PTX source in a string to the link"""

    def add_cu(self, cu, name):
        """Add CUDA source in a string to the link. The name of the source
        file should be specified in `name`."""
        with driver.get_active_context() as ac:
            dev = driver.get_device(ac.devnum)
            cc = dev.compute_capability
        ptx, log = nvrtc.compile(cu, name, cc)
        if config.DUMP_ASSEMBLY:
            print(('ASSEMBLY %s' % name).center(80, '-'))
            print(ptx)
            print('=' * 80)
        ptx_name = os.path.splitext(name)[0] + '.ptx'
        self.add_ptx(ptx.encode(), ptx_name)

    @abstractmethod
    def add_file(self, path, kind):
        """Add code from a file to the link"""

    def add_cu_file(self, path):
        with open(path, 'rb') as f:
            cu = f.read()
        self.add_cu(cu, os.path.basename(path))

    def add_file_guess_ext(self, path):
        """Add a file to the link, guessing its type from its extension."""
        ext = os.path.splitext(path)[1][1:]
        if ext == '':
            raise RuntimeError("Don't know how to link file with no extension")
        elif ext == 'cu':
            self.add_cu_file(path)
        else:
            kind = FILE_EXTENSION_MAP.get(ext, None)
            if kind is None:
                raise RuntimeError(f"Don't know how to link file with extension .{ext}")
            self.add_file(path, kind)

    @abstractmethod
    def complete(self):
        """Complete the link. Returns (cubin, size)

        cubin is a pointer to a internal buffer of cubin owned by the linker;
        thus, it should be loaded before the linker is destroyed.
        """