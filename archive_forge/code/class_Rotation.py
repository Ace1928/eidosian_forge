import xcffib
import struct
import io
from . import xproto
from . import render
class Rotation:
    Rotate_0 = 1 << 0
    Rotate_90 = 1 << 1
    Rotate_180 = 1 << 2
    Rotate_270 = 1 << 3
    Reflect_X = 1 << 4
    Reflect_Y = 1 << 5