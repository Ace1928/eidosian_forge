from typing import Any, Mapping
import warnings
import cupy
from cupy_backends.cuda.api import runtime
from cupy.cuda import device
from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
from cupyx.jit._internal_types import BuiltinFunc
from cupyx.jit._internal_types import Data
from cupyx.jit._internal_types import Constant
from cupyx.jit._internal_types import Range
from cupyx.jit import _compile
from functools import reduce
class SyncThreads(BuiltinFunc):

    def __call__(self):
        """Calls ``__syncthreads()``.

        .. seealso:: `Synchronization functions`_

        .. _Synchronization functions:
            https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions
        """
        super().__call__()

    def call_const(self, env):
        return Data('__syncthreads()', _cuda_types.void)