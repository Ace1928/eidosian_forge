import xcffib
import struct
import io
from . import xproto
from . import randr
from . import xfixes
from . import sync
class Capability:
    _None = 0
    Async = 1 << 0
    Fence = 1 << 1
    UST = 1 << 2
    AsyncMayTear = 1 << 3