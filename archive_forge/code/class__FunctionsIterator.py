from ctypes import (c_char_p, byref, POINTER, c_bool, create_string_buffer,
from llvmlite.binding import ffi
from llvmlite.binding.linker import link_modules
from llvmlite.binding.common import _decode_string, _encode_string
from llvmlite.binding.value import ValueRef, TypeRef
from llvmlite.binding.context import get_global_context
class _FunctionsIterator(_Iterator):
    kind = 'function'

    def _dispose(self):
        self._capi.LLVMPY_DisposeFunctionsIter(self)

    def _next(self):
        return ffi.lib.LLVMPY_FunctionsIterNext(self)