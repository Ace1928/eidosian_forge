import os
import sys
import warnings
from threading import RLock as threadRLock
from ctypes import CFUNCTYPE, c_int, CDLL, POINTER, c_uint
import numpy as np
import llvmlite.binding as ll
from llvmlite import ir
from numba.np.numpy_support import as_dtype
from numba.core import types, cgutils, config, errors
from numba.core.typing import signature
from numba.np.ufunc.wrappers import _wrapper_info
from numba.np.ufunc import ufuncbuilder
from numba.extending import overload, intrinsic
@intrinsic
def _iget_thread_id(typingctx):

    def codegen(context, builder, signature, args):
        mod = builder.module
        fnty = ir.FunctionType(cgutils.intp_t, [])
        fn = cgutils.get_or_insert_function(mod, fnty, 'get_thread_id')
        return builder.call(fn, [])
    return (signature(types.intp), codegen)