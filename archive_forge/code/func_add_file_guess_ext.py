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