from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
def _bytes_item(x):
    return ord(x)