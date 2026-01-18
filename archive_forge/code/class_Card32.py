from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class Card32(ValueField):
    structcode = 'L'
    structvalues = 1