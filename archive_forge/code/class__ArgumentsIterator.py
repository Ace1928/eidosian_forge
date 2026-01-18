from ctypes import (POINTER, byref, cast, c_char_p, c_double, c_int, c_size_t,
import enum
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
from llvmlite.binding.typeref import TypeRef
class _ArgumentsIterator(_ValueIterator):
    kind = 'argument'

    def _dispose(self):
        self._capi.LLVMPY_DisposeArgumentsIter(self)

    def _next(self):
        return ffi.lib.LLVMPY_ArgumentsIterNext(self)