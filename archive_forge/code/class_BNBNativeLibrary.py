import ctypes as ct
import logging
import os
from pathlib import Path
import torch
from bitsandbytes.consts import DYNAMIC_LIBRARY_SUFFIX, PACKAGE_DIR
from bitsandbytes.cuda_specs import CUDASpecs, get_cuda_specs
class BNBNativeLibrary:
    _lib: ct.CDLL
    compiled_with_cuda = False

    def __init__(self, lib: ct.CDLL):
        self._lib = lib

    def __getattr__(self, item):
        return getattr(self._lib, item)