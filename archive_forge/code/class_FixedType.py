from collections import deque
from enum import Enum, IntEnum, IntFlag
import struct
from typing import Optional
class FixedType:

    def __init__(self, size, struct_code):
        self.size = self.alignment = size
        self.struct_code = struct_code

    def parse_data(self, buf, pos, endianness, fds=()):
        pos += padding(pos, self.alignment)
        code = endianness.struct_code() + self.struct_code
        val = struct.unpack(code, buf[pos:pos + self.size])[0]
        return (val, pos + self.size)

    def serialise(self, data, pos, endianness, fds=None):
        pad = b'\x00' * padding(pos, self.alignment)
        code = endianness.struct_code() + self.struct_code
        return pad + struct.pack(code, data)

    def __repr__(self):
        return 'FixedType({!r}, {!r})'.format(self.size, self.struct_code)

    def __eq__(self, other):
        return type(other) is FixedType and self.size == other.size and (self.struct_code == other.struct_code)