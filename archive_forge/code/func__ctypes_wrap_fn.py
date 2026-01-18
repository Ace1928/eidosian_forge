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
def _ctypes_wrap_fn(self, fname, libfn=None):
    if libfn is None:
        try:
            proto = API_PROTOTYPES[fname]
        except KeyError:
            raise AttributeError(fname)
        restype = proto[0]
        argtypes = proto[1:]
        libfn = self._find_api(fname)
        libfn.restype = restype
        libfn.argtypes = argtypes

    def verbose_cuda_api_call(*args):
        argstr = ', '.join([str(arg) for arg in args])
        _logger.debug('call driver api: %s(%s)', libfn.__name__, argstr)
        retcode = libfn(*args)
        self._check_ctypes_error(fname, retcode)

    def safe_cuda_api_call(*args):
        _logger.debug('call driver api: %s', libfn.__name__)
        retcode = libfn(*args)
        self._check_ctypes_error(fname, retcode)
    if config.CUDA_LOG_API_ARGS:
        wrapper = verbose_cuda_api_call
    else:
        wrapper = safe_cuda_api_call
    safe_call = functools.wraps(libfn)(wrapper)
    setattr(self, fname, safe_call)
    return safe_call