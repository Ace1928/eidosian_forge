import xcffib
import struct
import io
from . import xfixes
from . import xproto
class DeviceUse:
    IsXPointer = 0
    IsXKeyboard = 1
    IsXExtensionDevice = 2
    IsXExtensionKeyboard = 3
    IsXExtensionPointer = 4