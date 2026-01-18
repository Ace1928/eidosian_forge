import ctypes as ct
import logging
import os
from pathlib import Path
import torch
from bitsandbytes.consts import DYNAMIC_LIBRARY_SUFFIX, PACKAGE_DIR
from bitsandbytes.cuda_specs import CUDASpecs, get_cuda_specs
class CudaBNBNativeLibrary(BNBNativeLibrary):
    compiled_with_cuda = True

    def __init__(self, lib: ct.CDLL):
        super().__init__(lib)
        lib.get_context.restype = ct.c_void_p
        lib.get_cusparse.restype = ct.c_void_p
        lib.cget_managed_ptr.restype = ct.c_void_p