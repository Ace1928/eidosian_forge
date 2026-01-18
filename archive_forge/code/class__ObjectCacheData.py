import platform
from ctypes import (POINTER, c_char_p, c_bool, c_void_p,
from llvmlite.binding import ffi, targets, object_file
class _ObjectCacheData(Structure):
    _fields_ = [('module_ptr', ffi.LLVMModuleRef), ('buf_ptr', c_void_p), ('buf_len', c_size_t)]