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
class MinFunc(BuiltinFunc):

    def call(self, env, *args, **kwds):
        if len(args) < 2:
            raise TypeError(f'min() expects at least 2 arguments, got {len(args)}')
        if kwds:
            raise TypeError('keyword arguments are not supported')
        return reduce(lambda a, b: _compile._call_ufunc(cupy.minimum, (a, b), None, env), args)