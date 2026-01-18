import xcffib
import struct
import io
from . import xproto
class CP:
    Repeat = 1 << 0
    AlphaMap = 1 << 1
    AlphaXOrigin = 1 << 2
    AlphaYOrigin = 1 << 3
    ClipXOrigin = 1 << 4
    ClipYOrigin = 1 << 5
    ClipMask = 1 << 6
    GraphicsExposure = 1 << 7
    SubwindowMode = 1 << 8
    PolyEdge = 1 << 9
    PolyMode = 1 << 10
    Dither = 1 << 11
    ComponentAlpha = 1 << 12