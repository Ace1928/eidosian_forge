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
class ParallelGUFuncBuilder(ufuncbuilder.GUFuncBuilder):

    def __init__(self, py_func, signature, identity=None, cache=False, targetoptions={}, writable_args=()):
        targetoptions.update(dict(nopython=True))
        super(ParallelGUFuncBuilder, self).__init__(py_func=py_func, signature=signature, identity=identity, cache=cache, targetoptions=targetoptions, writable_args=writable_args)

    def build(self, cres):
        """
        Returns (dtype numbers, function ptr, EnvironmentObject)
        """
        _launch_threads()
        info = build_gufunc_wrapper(self.py_func, cres, self.sin, self.sout, cache=self.cache, is_parfors=False)
        ptr = info.library.get_pointer_to_function(info.name)
        env = info.env
        dtypenums = []
        for a in cres.signature.args:
            if isinstance(a, types.Array):
                ty = a.dtype
            else:
                ty = a
            dtypenums.append(as_dtype(ty).num)
        return (dtypenums, ptr, env)