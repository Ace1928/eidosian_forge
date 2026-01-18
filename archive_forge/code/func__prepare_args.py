import numpy as np
import os
import sys
import ctypes
import functools
from numba.core import config, serialize, sigutils, types, typing, utils
from numba.core.caching import Cache, CacheImpl
from numba.core.compiler_lock import global_compiler_lock
from numba.core.dispatcher import Dispatcher
from numba.core.errors import NumbaPerformanceWarning
from numba.core.typing.typeof import Purpose, typeof
from numba.cuda.api import get_current_device
from numba.cuda.args import wrap_arg
from numba.cuda.compiler import compile_cuda, CUDACompiler
from numba.cuda.cudadrv import driver
from numba.cuda.cudadrv.devices import get_context
from numba.cuda.descriptor import cuda_target
from numba.cuda.errors import (missing_launch_config_msg,
from numba.cuda import types as cuda_types
from numba import cuda
from numba import _dispatcher
from warnings import warn
def _prepare_args(self, ty, val, stream, retr, kernelargs):
    """
        Convert arguments to ctypes and append to kernelargs
        """
    for extension in reversed(self.extensions):
        ty, val = extension.prepare_args(ty, val, stream=stream, retr=retr)
    if isinstance(ty, types.Array):
        devary = wrap_arg(val).to_device(retr, stream)
        c_intp = ctypes.c_ssize_t
        meminfo = ctypes.c_void_p(0)
        parent = ctypes.c_void_p(0)
        nitems = c_intp(devary.size)
        itemsize = c_intp(devary.dtype.itemsize)
        ptr = driver.device_pointer(devary)
        if driver.USE_NV_BINDING:
            ptr = int(ptr)
        data = ctypes.c_void_p(ptr)
        kernelargs.append(meminfo)
        kernelargs.append(parent)
        kernelargs.append(nitems)
        kernelargs.append(itemsize)
        kernelargs.append(data)
        for ax in range(devary.ndim):
            kernelargs.append(c_intp(devary.shape[ax]))
        for ax in range(devary.ndim):
            kernelargs.append(c_intp(devary.strides[ax]))
    elif isinstance(ty, types.Integer):
        cval = getattr(ctypes, 'c_%s' % ty)(val)
        kernelargs.append(cval)
    elif ty == types.float16:
        cval = ctypes.c_uint16(np.float16(val).view(np.uint16))
        kernelargs.append(cval)
    elif ty == types.float64:
        cval = ctypes.c_double(val)
        kernelargs.append(cval)
    elif ty == types.float32:
        cval = ctypes.c_float(val)
        kernelargs.append(cval)
    elif ty == types.boolean:
        cval = ctypes.c_uint8(int(val))
        kernelargs.append(cval)
    elif ty == types.complex64:
        kernelargs.append(ctypes.c_float(val.real))
        kernelargs.append(ctypes.c_float(val.imag))
    elif ty == types.complex128:
        kernelargs.append(ctypes.c_double(val.real))
        kernelargs.append(ctypes.c_double(val.imag))
    elif isinstance(ty, (types.NPDatetime, types.NPTimedelta)):
        kernelargs.append(ctypes.c_int64(val.view(np.int64)))
    elif isinstance(ty, types.Record):
        devrec = wrap_arg(val).to_device(retr, stream)
        ptr = devrec.device_ctypes_pointer
        if driver.USE_NV_BINDING:
            ptr = ctypes.c_void_p(int(ptr))
        kernelargs.append(ptr)
    elif isinstance(ty, types.BaseTuple):
        assert len(ty) == len(val)
        for t, v in zip(ty, val):
            self._prepare_args(t, v, stream, retr, kernelargs)
    elif isinstance(ty, types.EnumMember):
        try:
            self._prepare_args(ty.dtype, val.value, stream, retr, kernelargs)
        except NotImplementedError:
            raise NotImplementedError(ty, val)
    else:
        raise NotImplementedError(ty, val)