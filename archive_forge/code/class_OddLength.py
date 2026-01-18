from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class OddLength(LengthField):
    structcode = 'B'
    structvalues = 1

    def __init__(self, name):
        self.name = name

    def calc_length(self, length):
        return length % 2

    def parse_value(self, value, display):
        if value == 0:
            return 'even'
        else:
            return 'odd'