from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class ResourceObj:
    structcode = 'L'
    structvalues = 1

    def __init__(self, class_name):
        self.class_name = class_name

    def parse_value(self, value, display):
        c = display.get_resource_class(self.class_name)
        if c:
            return c(display, value)
        else:
            return value