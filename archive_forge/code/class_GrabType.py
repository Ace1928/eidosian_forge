import xcffib
import struct
import io
from . import xfixes
from . import xproto
class GrabType:
    Button = 0
    Keycode = 1
    Enter = 2
    FocusIn = 3
    TouchBegin = 4
    GesturePinchBegin = 5
    GestureSwipeBegin = 6