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
def _find_api(self, fname):
    if config.CUDA_PER_THREAD_DEFAULT_STREAM and (not USE_NV_BINDING):
        variants = ('_v2_ptds', '_v2_ptsz', '_ptds', '_ptsz', '_v2', '')
    else:
        variants = ('_v2', '')
    for variant in variants:
        try:
            return getattr(self.lib, f'{fname}{variant}')
        except AttributeError:
            pass

    def absent_function(*args, **kws):
        raise CudaDriverError(f'Driver missing function: {fname}')
    setattr(self, fname, absent_function)
    return absent_function