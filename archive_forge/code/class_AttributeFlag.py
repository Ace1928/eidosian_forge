import xcffib
import struct
import io
from . import xproto
from . import shm
class AttributeFlag:
    Gettable = 1 << 0
    Settable = 1 << 1