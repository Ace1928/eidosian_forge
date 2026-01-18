from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class Int16(ValueField):
    structcode = 'h'
    structvalues = 1