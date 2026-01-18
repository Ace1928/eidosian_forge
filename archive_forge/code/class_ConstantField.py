from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class ConstantField(Field):

    def __init__(self, value):
        self.value = value