from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class RequestLength(TotalLengthField):
    structcode = 'H'
    structvalues = 1

    def calc_length(self, length):
        return length // 4