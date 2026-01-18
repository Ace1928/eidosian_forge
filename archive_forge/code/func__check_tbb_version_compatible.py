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
def _check_tbb_version_compatible():
    """
    Checks that if TBB is present it is of a compatible version.
    """
    try:
        if _IS_WINDOWS:
            libtbb_name = 'tbb12.dll'
        elif _IS_OSX:
            libtbb_name = 'libtbb.12.dylib'
        elif _IS_LINUX:
            libtbb_name = 'libtbb.so.12'
        else:
            raise ValueError('Unknown operating system')
        libtbb = CDLL(libtbb_name)
        version_func = libtbb.TBB_runtime_interface_version
        version_func.argtypes = []
        version_func.restype = c_int
        tbb_iface_ver = version_func()
        if tbb_iface_ver < 12060:
            msg = 'The TBB threading layer requires TBB version 2021 update 6 or later i.e., TBB_INTERFACE_VERSION >= 12060. Found TBB_INTERFACE_VERSION = %s. The TBB threading layer is disabled.' % tbb_iface_ver
            problem = errors.NumbaWarning(msg)
            warnings.warn(problem)
            raise ImportError('Problem with TBB. Reason: %s' % msg)
    except (ValueError, OSError) as e:
        raise ImportError('Problem with TBB. Reason: %s' % e)