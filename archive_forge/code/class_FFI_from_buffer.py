from types import BuiltinFunctionType
import ctypes
from functools import partial
import numpy as np
from numba.core import types
from numba.core.errors import TypingError
from numba.core.typing import templates
from numba.np import numpy_support
@registry.register
class FFI_from_buffer(templates.AbstractTemplate):
    key = 'ffi.from_buffer'

    def generic(self, args, kws):
        if kws or len(args) != 1:
            return
        [ary] = args
        if not isinstance(ary, types.Buffer):
            raise TypingError('from_buffer() expected a buffer object, got %s' % (ary,))
        if ary.layout not in ('C', 'F'):
            raise TypingError('from_buffer() unsupported on non-contiguous buffers (got %s)' % (ary,))
        if ary.layout != 'C' and ary.ndim > 1:
            raise TypingError('from_buffer() only supports multidimensional arrays with C layout (got %s)' % (ary,))
        ptr = types.CPointer(ary.dtype)
        return templates.signature(ptr, ary)