import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
class BarrierDirections:
    PositiveX = 1 << 0
    PositiveY = 1 << 1
    NegativeX = 1 << 2
    NegativeY = 1 << 3