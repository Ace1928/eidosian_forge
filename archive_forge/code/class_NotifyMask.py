import xcffib
import struct
import io
from . import xproto
from . import render
class NotifyMask:
    ScreenChange = 1 << 0
    CrtcChange = 1 << 1
    OutputChange = 1 << 2
    OutputProperty = 1 << 3
    ProviderChange = 1 << 4
    ProviderProperty = 1 << 5
    ResourceChange = 1 << 6
    Lease = 1 << 7