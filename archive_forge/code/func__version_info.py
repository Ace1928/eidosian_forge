from ctypes import c_uint
from llvmlite.binding import ffi
def _version_info():
    v = []
    x = ffi.lib.LLVMPY_GetVersionInfo()
    while x:
        v.append(x & 255)
        x >>= 8
    return tuple(reversed(v))