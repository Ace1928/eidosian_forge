from ctypes import (POINTER, byref, cast, c_char_p, c_double, c_int, c_size_t,
import enum
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
from llvmlite.binding.typeref import TypeRef
def add_function_attribute(self, attr):
    """Only works on function value

        Parameters
        -----------
        attr : str
            attribute name
        """
    if not self.is_function:
        raise ValueError('expected function value, got %s' % (self._kind,))
    attrname = str(attr)
    attrval = ffi.lib.LLVMPY_GetEnumAttributeKindForName(_encode_string(attrname), len(attrname))
    if attrval == 0:
        raise ValueError('no such attribute {!r}'.format(attrname))
    ffi.lib.LLVMPY_AddFunctionAttr(self, attrval)