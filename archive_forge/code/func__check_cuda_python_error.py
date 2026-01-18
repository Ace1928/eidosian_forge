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
def _check_cuda_python_error(self, fname, returned):
    retcode = returned[0]
    retval = returned[1:]
    if len(retval) == 1:
        retval = retval[0]
    if retcode != binding.CUresult.CUDA_SUCCESS:
        msg = 'Call to %s results in %s' % (fname, retcode.name)
        _logger.error(msg)
        if retcode == binding.CUresult.CUDA_ERROR_NOT_INITIALIZED:
            self._detect_fork()
        raise CudaAPIError(retcode, msg)
    return retval