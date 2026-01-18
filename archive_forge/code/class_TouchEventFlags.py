import xcffib
import struct
import io
from . import xfixes
from . import xproto
class TouchEventFlags:
    TouchPendingEnd = 1 << 16
    TouchEmulatingPointer = 1 << 17