import functools
import warnings
import numpy as np
from numba import jit, typeof
from numba.core import cgutils, types, serialize, sigutils, errors
from numba.core.extending import (is_jitted, overload_attribute,
from numba.core.typing import npydecl
from numba.core.typing.templates import AbstractTemplate, signature
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.np.ufunc import _internal
from numba.parfors import array_analysis
from numba.np.ufunc import ufuncbuilder
from numba.np import numpy_support
from typing import Callable
from llvmlite import ir
class DUFuncLowerer(object):
    """Callable class responsible for lowering calls to a specific DUFunc.
    """

    def __init__(self, dufunc):
        self.kernel = make_dufunc_kernel(dufunc)
        self.libs = []

    def __call__(self, context, builder, sig, args):
        from numba.np import npyimpl
        return npyimpl.numpy_ufunc_kernel(context, builder, sig, args, self.kernel.dufunc.ufunc, self.kernel)