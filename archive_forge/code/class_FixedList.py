from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class FixedList(List):

    def __init__(self, name, size, type, pad=1):
        List.__init__(self, name, type, pad)
        self.size = size

    def parse_binary_value(self, data, display, length, format):
        return List.parse_binary_value(self, data, display, self.size, format)

    def pack_value(self, val):
        if len(val) != self.size:
            raise BadDataError('length mismatch for FixedList %s' % self.name)
        return List.pack_value(self, val)