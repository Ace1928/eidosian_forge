import xcffib
import struct
import io
from . import xproto
from . import shm
class ScanlineOrder:
    TopToBottom = 0
    BottomToTop = 1