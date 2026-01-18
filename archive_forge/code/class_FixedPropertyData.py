from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class FixedPropertyData(PropertyData):

    def __init__(self, name, size):
        PropertyData.__init__(self, name)
        self.size = size

    def parse_binary_value(self, data, display, length, format):
        return PropertyData.parse_binary_value(self, data, display, self.size // (format // 8), format)

    def pack_value(self, value):
        data, dlen, fmt = PropertyData.pack_value(self, value)
        if len(data) != self.size:
            raise BadDataError('Wrong data length for FixedPropertyData: %s' % (value,))
        return (data, dlen, fmt)