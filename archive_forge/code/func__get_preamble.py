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
def _get_preamble(self):
    preamble = '__device__ __forceinline__ unsigned int LaneId() {'
    if not runtime.is_hip:
        preamble += '\n                unsigned int ret;\n                asm ("mov.u32 %0, %%laneid;" : "=r"(ret) );\n                return ret; }\n            '
    else:
        preamble += '\n                return __lane_id(); }\n            '
    return preamble