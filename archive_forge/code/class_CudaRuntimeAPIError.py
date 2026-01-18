import ctypes
import functools
import sys
from numba.core import config
from numba.cuda.cudadrv.driver import ERROR_MAP, make_logger
from numba.cuda.cudadrv.error import CudaSupportError, CudaRuntimeError
from numba.cuda.cudadrv.libs import open_cudalib
from numba.cuda.cudadrv.rtapi import API_PROTOTYPES
from numba.cuda.cudadrv import enums
class CudaRuntimeAPIError(CudaRuntimeError):
    """
    Raised when there is an error accessing a C API from the CUDA Runtime.
    """

    def __init__(self, code, msg):
        self.code = code
        self.msg = msg
        super().__init__(code, msg)

    def __str__(self):
        return '[%s] %s' % (self.code, self.msg)