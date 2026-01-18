import sys
from ctypes import *
class _swapped_meta:

    def __setattr__(self, attrname, value):
        if attrname == '_fields_':
            fields = []
            for desc in value:
                name = desc[0]
                typ = desc[1]
                rest = desc[2:]
                fields.append((name, _other_endian(typ)) + rest)
            value = fields
        super().__setattr__(attrname, value)