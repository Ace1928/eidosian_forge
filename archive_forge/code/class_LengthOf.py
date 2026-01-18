from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class LengthOf(LengthField):

    def __init__(self, name, size):
        self.name = name
        self.structcode = unsigned_codes[size]