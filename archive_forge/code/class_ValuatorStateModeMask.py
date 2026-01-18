import xcffib
import struct
import io
from . import xfixes
from . import xproto
class ValuatorStateModeMask:
    DeviceModeAbsolute = 1 << 0
    OutOfProximity = 1 << 1