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
class CtypesLinker(Linker):
    """
    Links for current device if no CC given
    """

    def __init__(self, max_registers=0, lineinfo=False, cc=None):
        logsz = config.CUDA_LOG_SIZE
        linkerinfo = (c_char * logsz)()
        linkererrors = (c_char * logsz)()
        options = {enums.CU_JIT_INFO_LOG_BUFFER: addressof(linkerinfo), enums.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: c_void_p(logsz), enums.CU_JIT_ERROR_LOG_BUFFER: addressof(linkererrors), enums.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_void_p(logsz), enums.CU_JIT_LOG_VERBOSE: c_void_p(1)}
        if max_registers:
            options[enums.CU_JIT_MAX_REGISTERS] = c_void_p(max_registers)
        if lineinfo:
            options[enums.CU_JIT_GENERATE_LINE_INFO] = c_void_p(1)
        if cc is None:
            options[enums.CU_JIT_TARGET_FROM_CUCONTEXT] = 1
        else:
            cc_val = cc[0] * 10 + cc[1]
            options[enums.CU_JIT_TARGET] = c_void_p(cc_val)
        raw_keys = list(options.keys())
        raw_values = list(options.values())
        option_keys = (drvapi.cu_jit_option * len(raw_keys))(*raw_keys)
        option_vals = (c_void_p * len(raw_values))(*raw_values)
        self.handle = handle = drvapi.cu_link_state()
        driver.cuLinkCreate(len(raw_keys), option_keys, option_vals, byref(self.handle))
        weakref.finalize(self, driver.cuLinkDestroy, handle)
        self.linker_info_buf = linkerinfo
        self.linker_errors_buf = linkererrors
        self._keep_alive = [linkerinfo, linkererrors, option_keys, option_vals]

    @property
    def info_log(self):
        return self.linker_info_buf.value.decode('utf8')

    @property
    def error_log(self):
        return self.linker_errors_buf.value.decode('utf8')

    def add_ptx(self, ptx, name='<cudapy-ptx>'):
        ptxbuf = c_char_p(ptx)
        namebuf = c_char_p(name.encode('utf8'))
        self._keep_alive += [ptxbuf, namebuf]
        try:
            driver.cuLinkAddData(self.handle, enums.CU_JIT_INPUT_PTX, ptxbuf, len(ptx), namebuf, 0, None, None)
        except CudaAPIError as e:
            raise LinkerError('%s\n%s' % (e, self.error_log))

    def add_file(self, path, kind):
        pathbuf = c_char_p(path.encode('utf8'))
        self._keep_alive.append(pathbuf)
        try:
            driver.cuLinkAddFile(self.handle, kind, pathbuf, 0, None, None)
        except CudaAPIError as e:
            if e.code == enums.CUDA_ERROR_FILE_NOT_FOUND:
                msg = f'{path} not found'
            else:
                msg = '%s\n%s' % (e, self.error_log)
            raise LinkerError(msg)

    def complete(self):
        cubin_buf = c_void_p(0)
        size = c_size_t(0)
        try:
            driver.cuLinkComplete(self.handle, byref(cubin_buf), byref(size))
        except CudaAPIError as e:
            raise LinkerError('%s\n%s' % (e, self.error_log))
        size = size.value
        assert size > 0, 'linker returned a zero sized cubin'
        del self._keep_alive[:]
        cubin_ptr = ctypes.cast(cubin_buf, ctypes.POINTER(ctypes.c_char))
        return bytes(np.ctypeslib.as_array(cubin_ptr, shape=(size,)))