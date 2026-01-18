import xcffib
import struct
import io
from . import xfixes
from . import xproto
class HierarchyMask:
    MasterAdded = 1 << 0
    MasterRemoved = 1 << 1
    SlaveAdded = 1 << 2
    SlaveRemoved = 1 << 3
    SlaveAttached = 1 << 4
    SlaveDetached = 1 << 5
    DeviceEnabled = 1 << 6
    DeviceDisabled = 1 << 7