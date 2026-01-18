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
def _ctypes_max_potential_block_size(self, func, b2d_func, memsize, blocksizelimit, flags):
    gridsize = c_int()
    blocksize = c_int()
    b2d_cb = cu_occupancy_b2d_size(b2d_func)
    args = [byref(gridsize), byref(blocksize), func.handle, b2d_cb, memsize, blocksizelimit]
    if not flags:
        driver.cuOccupancyMaxPotentialBlockSize(*args)
    else:
        args.append(flags)
        driver.cuOccupancyMaxPotentialBlockSizeWithFlags(*args)
    return (gridsize.value, blocksize.value)