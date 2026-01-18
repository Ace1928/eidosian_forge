from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class Card8(ValueField):
    structcode = 'B'
    structvalues = 1