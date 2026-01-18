import functools
import warnings
import numpy
import cupy
from cupy._core import core
from cupyx.jit import _compile
from cupyx.jit import _cuda_typerules
from cupyx.jit import _cuda_types
from cupyx.jit import _internal_types
from cupyx.jit._cuda_types import Scalar
def _emit_code_from_types(self, in_types, ret_type=None):
    return _compile.transpile(self.func, self.attributes, self.mode, in_types, ret_type)