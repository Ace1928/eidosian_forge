from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class String16(ValueField):
    structcode = None

    def __init__(self, name, pad=1):
        ValueField.__init__(self, name)
        self.pad = pad

    def pack_value(self, val):
        if type(val) is str:
            val = [ord(c) for c in val]
        slen = len(val)
        if self.pad:
            pad = b'\x00\x00' * (slen % 2)
        else:
            pad = b''
        return (struct.pack(*('>' + 'H' * slen,) + tuple(val)) + pad, slen, None)

    def parse_binary_value(self, data, display, length, format):
        if length == 'odd':
            length = len(data) // 2 - 1
        elif length == 'even':
            length = len(data) // 2
        if self.pad:
            slen = length + length % 2
        else:
            slen = length
        return (struct.unpack('>' + 'H' * length, data[:length * 2]), data[slen * 2:])