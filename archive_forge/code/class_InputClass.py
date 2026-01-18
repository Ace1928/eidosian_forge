import xcffib
import struct
import io
from . import xfixes
from . import xproto
class InputClass:
    Key = 0
    Button = 1
    Valuator = 2
    Feedback = 3
    Proximity = 4
    Focus = 5
    Other = 6