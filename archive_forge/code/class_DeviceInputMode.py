import xcffib
import struct
import io
from . import xfixes
from . import xproto
class DeviceInputMode:
    AsyncThisDevice = 0
    SyncThisDevice = 1
    ReplayThisDevice = 2
    AsyncOtherDevices = 3
    AsyncAll = 4
    SyncAll = 5