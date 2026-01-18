import os
from ctypes import (POINTER, c_char_p, c_longlong, c_int, c_size_t,
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
class TargetData(ffi.ObjectRef):
    """
    A TargetData provides structured access to a data layout.
    Use :func:`create_target_data` to create instances.
    """

    def __str__(self):
        if self._closed:
            return '<dead TargetData>'
        with ffi.OutputString() as out:
            ffi.lib.LLVMPY_CopyStringRepOfTargetData(self, out)
            return str(out)

    def _dispose(self):
        self._capi.LLVMPY_DisposeTargetData(self)

    def get_abi_size(self, ty):
        """
        Get ABI size of LLVM type *ty*.
        """
        return ffi.lib.LLVMPY_ABISizeOfType(self, ty)

    def get_element_offset(self, ty, position):
        """
        Get byte offset of type's ty element at the given position
        """
        offset = ffi.lib.LLVMPY_OffsetOfElement(self, ty, position)
        if offset == -1:
            raise ValueError("Could not determined offset of {}th element of the type '{}'. Is it a structtype?".format(position, str(ty)))
        return offset

    def get_pointee_abi_size(self, ty):
        """
        Get ABI size of pointee type of LLVM pointer type *ty*.
        """
        size = ffi.lib.LLVMPY_ABISizeOfElementType(self, ty)
        if size == -1:
            raise RuntimeError('Not a pointer type: %s' % (ty,))
        return size

    def get_pointee_abi_alignment(self, ty):
        """
        Get minimum ABI alignment of pointee type of LLVM pointer type *ty*.
        """
        size = ffi.lib.LLVMPY_ABIAlignmentOfElementType(self, ty)
        if size == -1:
            raise RuntimeError('Not a pointer type: %s' % (ty,))
        return size