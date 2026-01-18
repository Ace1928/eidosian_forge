import xcffib
import struct
import io
from . import xproto
from . import xfixes
class ReportLevel:
    RawRectangles = 0
    DeltaRectangles = 1
    BoundingBox = 2
    NonEmpty = 3