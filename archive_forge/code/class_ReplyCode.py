from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class ReplyCode(ConstantField):
    structcode = 'B'
    structvalues = 1

    def __init__(self):
        self.value = 1