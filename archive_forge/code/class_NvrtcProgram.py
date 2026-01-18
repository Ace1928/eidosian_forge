from ctypes import byref, c_char, c_char_p, c_int, c_size_t, c_void_p, POINTER
from enum import IntEnum
from numba.core import config
from numba.cuda.cudadrv.error import (NvrtcError, NvrtcCompilationError,
import functools
import os
import threading
import warnings
class NvrtcProgram:
    """
    A class for managing the lifetime of nvrtcProgram instances. Instances of
    the class own an nvrtcProgram; when an instance is deleted, the underlying
    nvrtcProgram is destroyed using the appropriate NVRTC API.
    """

    def __init__(self, nvrtc, handle):
        self._nvrtc = nvrtc
        self._handle = handle

    @property
    def handle(self):
        return self._handle

    def __del__(self):
        if self._handle:
            self._nvrtc.destroy_program(self)