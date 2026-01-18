from ctypes import (c_char_p, byref, POINTER, c_bool, create_string_buffer,
from llvmlite.binding import ffi
from llvmlite.binding.linker import link_modules
from llvmlite.binding.common import _decode_string, _encode_string
from llvmlite.binding.value import ValueRef, TypeRef
from llvmlite.binding.context import get_global_context
class _TypesIterator(_Iterator):
    kind = 'type'

    def _dispose(self):
        self._capi.LLVMPY_DisposeTypesIter(self)

    def __next__(self):
        vp = self._next()
        if vp:
            return TypeRef(vp)
        else:
            raise StopIteration

    def _next(self):
        return ffi.lib.LLVMPY_TypesIterNext(self)
    next = __next__