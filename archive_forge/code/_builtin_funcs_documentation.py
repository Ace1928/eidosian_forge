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
Returns the lane ID of the calling thread, ranging in
        ``[0, jit.warpsize)``.

        .. note::
            Unlike :obj:`numba.cuda.laneid`, this is a callable function
            instead of a property.
        