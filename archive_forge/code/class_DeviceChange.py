import xcffib
import struct
import io
from . import xfixes
from . import xproto
class DeviceChange:
    Added = 0
    Removed = 1
    Enabled = 2
    Disabled = 3
    Unrecoverable = 4
    ControlChanged = 5