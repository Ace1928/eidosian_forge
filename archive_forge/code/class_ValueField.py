from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class ValueField(Field):

    def __init__(self, name, default=None):
        self.name = name
        self.default = default