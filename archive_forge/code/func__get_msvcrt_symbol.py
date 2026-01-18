import sys
from llvmlite import ir
import llvmlite.binding as ll
from numba.core import utils, intrinsics
from numba import _helperlib
def _get_msvcrt_symbol(symbol):
    """
    Under Windows, look up a symbol inside the C runtime
    and return the raw pointer value as an integer.
    """
    from ctypes import cdll, cast, c_void_p
    f = getattr(cdll.msvcrt, symbol)
    return cast(f, c_void_p).value