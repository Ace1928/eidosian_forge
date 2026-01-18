import xcffib
import struct
import io
from . import xfixes
from . import xproto
class DeviceClassType:
    Key = 0
    Button = 1
    Valuator = 2
    Scroll = 3
    Touch = 8
    Gesture = 9