from llvmlite.binding import ffi
from ctypes import (c_bool, c_char_p, c_char, c_size_t, string_at, c_uint64,
class ObjectFileRef(ffi.ObjectRef):

    @classmethod
    def from_data(cls, data):
        return cls(ffi.lib.LLVMPY_CreateObjectFile(data, len(data)))

    @classmethod
    def from_path(cls, path):
        with open(path, 'rb') as f:
            data = f.read()
        return cls(ffi.lib.LLVMPY_CreateObjectFile(data, len(data)))

    def sections(self):
        it = SectionIteratorRef(ffi.lib.LLVMPY_GetSections(self))
        while not it.is_end(self):
            yield it
            it.next()

    def _dispose(self):
        ffi.lib.LLVMPY_DisposeObjectFile(self)