from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class ScalarObj:

    def __init__(self, code):
        self.structcode = code
        self.structvalues = 1
        self.parse_value = None