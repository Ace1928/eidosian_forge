from llvmlite.binding import ffi
from ctypes import (c_bool, c_char_p, c_char, c_size_t, string_at, c_uint64,
class SectionIteratorRef(ffi.ObjectRef):

    def name(self):
        return ffi.lib.LLVMPY_GetSectionName(self)

    def is_text(self):
        return ffi.lib.LLVMPY_IsSectionText(self)

    def size(self):
        return ffi.lib.LLVMPY_GetSectionSize(self)

    def address(self):
        return ffi.lib.LLVMPY_GetSectionAddress(self)

    def data(self):
        return string_at(ffi.lib.LLVMPY_GetSectionContents(self), self.size())

    def is_end(self, object_file):
        return ffi.lib.LLVMPY_IsSectionIteratorAtEnd(object_file, self)

    def next(self):
        ffi.lib.LLVMPY_MoveToNextSection(self)

    def _dispose(self):
        ffi.lib.LLVMPY_DisposeSectionIterator(self)