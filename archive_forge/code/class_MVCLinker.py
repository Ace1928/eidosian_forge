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
class MVCLinker(Linker):
    """
    Linker supporting Minor Version Compatibility, backed by the cubinlinker
    package.
    """

    def __init__(self, max_registers=None, lineinfo=False, cc=None):
        try:
            from cubinlinker import CubinLinker
        except ImportError as err:
            raise ImportError(_MVC_ERROR_MESSAGE) from err
        if cc is None:
            raise RuntimeError('MVCLinker requires Compute Capability to be specified, but cc is None')
        arch = f'sm_{cc[0] * 10 + cc[1]}'
        ptx_compile_opts = ['--gpu-name', arch, '-c']
        if max_registers:
            arg = f'--maxrregcount={max_registers}'
            ptx_compile_opts.append(arg)
        if lineinfo:
            ptx_compile_opts.append('--generate-line-info')
        self.ptx_compile_options = tuple(ptx_compile_opts)
        self._linker = CubinLinker(f'--arch={arch}')

    @property
    def info_log(self):
        return self._linker.info_log

    @property
    def error_log(self):
        return self._linker.error_log

    def add_ptx(self, ptx, name='<cudapy-ptx>'):
        try:
            from ptxcompiler import compile_ptx
            from cubinlinker import CubinLinkerError
        except ImportError as err:
            raise ImportError(_MVC_ERROR_MESSAGE) from err
        compile_result = compile_ptx(ptx.decode(), self.ptx_compile_options)
        try:
            self._linker.add_cubin(compile_result.compiled_program, name)
        except CubinLinkerError as e:
            raise LinkerError from e

    def add_file(self, path, kind):
        try:
            from cubinlinker import CubinLinkerError
        except ImportError as err:
            raise ImportError(_MVC_ERROR_MESSAGE) from err
        try:
            with open(path, 'rb') as f:
                data = f.read()
        except FileNotFoundError:
            raise LinkerError(f'{path} not found')
        name = pathlib.Path(path).name
        if kind == FILE_EXTENSION_MAP['cubin']:
            fn = self._linker.add_cubin
        elif kind == FILE_EXTENSION_MAP['fatbin']:
            fn = self._linker.add_fatbin
        elif kind == FILE_EXTENSION_MAP['a']:
            raise LinkerError(f"Don't know how to link {kind}")
        elif kind == FILE_EXTENSION_MAP['ptx']:
            return self.add_ptx(data, name)
        else:
            raise LinkerError(f"Don't know how to link {kind}")
        try:
            fn(data, name)
        except CubinLinkerError as e:
            raise LinkerError from e

    def complete(self):
        try:
            from cubinlinker import CubinLinkerError
        except ImportError as err:
            raise ImportError(_MVC_ERROR_MESSAGE) from err
        try:
            return self._linker.complete()
        except CubinLinkerError as e:
            raise LinkerError from e