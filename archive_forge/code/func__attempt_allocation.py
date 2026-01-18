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
def _attempt_allocation(self, allocator):
    """
        Attempt allocation by calling *allocator*.  If an out-of-memory error
        is raised, the pending deallocations are flushed and the allocation
        is retried.  If it fails in the second attempt, the error is reraised.
        """
    try:
        return allocator()
    except CudaAPIError as e:
        if USE_NV_BINDING:
            oom_code = binding.CUresult.CUDA_ERROR_OUT_OF_MEMORY
        else:
            oom_code = enums.CUDA_ERROR_OUT_OF_MEMORY
        if e.code == oom_code:
            self.deallocations.clear()
            return allocator()
        else:
            raise