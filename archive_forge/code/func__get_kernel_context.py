from contextlib import contextmanager
import functools
import sys
import threading
import numpy as np
from .cudadrv.devicearray import FakeCUDAArray, FakeWithinKernelCUDAArray
from .kernelapi import Dim3, FakeCUDAModule, swapped_cuda_module
from ..errors import normalize_kernel_dimensions
from ..args import wrap_arg, ArgHint
def _get_kernel_context():
    """
    Get the current kernel context. This is usually done by a device function.
    """
    return _kernel_context