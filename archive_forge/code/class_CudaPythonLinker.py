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
class CudaPythonLinker(Linker):
    """
    Links for current device if no CC given
    """

    def __init__(self, max_registers=0, lineinfo=False, cc=None):
        logsz = config.CUDA_LOG_SIZE
        linkerinfo = bytearray(logsz)
        linkererrors = bytearray(logsz)
        jit_option = binding.CUjit_option
        options = {jit_option.CU_JIT_INFO_LOG_BUFFER: linkerinfo, jit_option.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: logsz, jit_option.CU_JIT_ERROR_LOG_BUFFER: linkererrors, jit_option.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: logsz, jit_option.CU_JIT_LOG_VERBOSE: 1}
        if max_registers:
            options[jit_option.CU_JIT_MAX_REGISTERS] = max_registers
        if lineinfo:
            options[jit_option.CU_JIT_GENERATE_LINE_INFO] = 1
        if cc is None:
            options[jit_option.CU_JIT_TARGET_FROM_CUCONTEXT] = 1
        else:
            cc_val = cc[0] * 10 + cc[1]
            cc_enum = getattr(binding.CUjit_target, f'CU_TARGET_COMPUTE_{cc_val}')
            options[jit_option.CU_JIT_TARGET] = cc_enum
        raw_keys = list(options.keys())
        raw_values = list(options.values())
        self.handle = driver.cuLinkCreate(len(raw_keys), raw_keys, raw_values)
        weakref.finalize(self, driver.cuLinkDestroy, self.handle)
        self.linker_info_buf = linkerinfo
        self.linker_errors_buf = linkererrors
        self._keep_alive = [linkerinfo, linkererrors, raw_keys, raw_values]

    @property
    def info_log(self):
        return self.linker_info_buf.decode('utf8')

    @property
    def error_log(self):
        return self.linker_errors_buf.decode('utf8')

    def add_ptx(self, ptx, name='<cudapy-ptx>'):
        namebuf = name.encode('utf8')
        self._keep_alive += [ptx, namebuf]
        try:
            input_ptx = binding.CUjitInputType.CU_JIT_INPUT_PTX
            driver.cuLinkAddData(self.handle, input_ptx, ptx, len(ptx), namebuf, 0, [], [])
        except CudaAPIError as e:
            raise LinkerError('%s\n%s' % (e, self.error_log))

    def add_file(self, path, kind):
        pathbuf = path.encode('utf8')
        self._keep_alive.append(pathbuf)
        try:
            driver.cuLinkAddFile(self.handle, kind, pathbuf, 0, [], [])
        except CudaAPIError as e:
            if e.code == binding.CUresult.CUDA_ERROR_FILE_NOT_FOUND:
                msg = f'{path} not found'
            else:
                msg = '%s\n%s' % (e, self.error_log)
            raise LinkerError(msg)

    def complete(self):
        try:
            cubin_buf, size = driver.cuLinkComplete(self.handle)
        except CudaAPIError as e:
            raise LinkerError('%s\n%s' % (e, self.error_log))
        assert size > 0, 'linker returned a zero sized cubin'
        del self._keep_alive[:]
        cubin_ptr = ctypes.cast(cubin_buf, ctypes.POINTER(ctypes.c_char))
        return bytes(np.ctypeslib.as_array(cubin_ptr, shape=(size,)))