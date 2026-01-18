import sys
import pickle
import struct
import pprint
import zipfile
import fnmatch
from typing import Any, IO, BinaryIO, Union
class DumpUnpickler(pickle._Unpickler):

    def __init__(self, file, *, catch_invalid_utf8=False, **kwargs):
        super().__init__(file, **kwargs)
        self.catch_invalid_utf8 = catch_invalid_utf8

    def find_class(self, module, name):
        return FakeClass(module, name)

    def persistent_load(self, pid):
        return FakeObject('pers', 'obj', (pid,))
    dispatch = dict(pickle._Unpickler.dispatch)

    def load_binunicode(self):
        strlen, = struct.unpack('<I', self.read(4))
        if strlen > sys.maxsize:
            raise Exception('String too long.')
        str_bytes = self.read(strlen)
        obj: Any
        try:
            obj = str(str_bytes, 'utf-8', 'surrogatepass')
        except UnicodeDecodeError as exn:
            if not self.catch_invalid_utf8:
                raise
            obj = FakeObject('builtin', 'UnicodeDecodeError', (str(exn),))
        self.append(obj)
    dispatch[pickle.BINUNICODE[0]] = load_binunicode

    @classmethod
    def dump(cls, in_stream, out_stream):
        value = cls(in_stream).load()
        pprint.pprint(value, stream=out_stream)
        return value