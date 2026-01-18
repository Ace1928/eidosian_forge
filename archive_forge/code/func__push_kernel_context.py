from contextlib import contextmanager
import functools
import sys
import threading
import numpy as np
from .cudadrv.devicearray import FakeCUDAArray, FakeWithinKernelCUDAArray
from .kernelapi import Dim3, FakeCUDAModule, swapped_cuda_module
from ..errors import normalize_kernel_dimensions
from ..args import wrap_arg, ArgHint
@contextmanager
def _push_kernel_context(mod):
    """
    Push the current kernel context.
    """
    global _kernel_context
    assert _kernel_context is None, 'concurrent simulated kernel not supported'
    _kernel_context = mod
    try:
        yield
    finally:
        _kernel_context = None