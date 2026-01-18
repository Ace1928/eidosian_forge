import ctypes
import ctypes.util
import sys
class PmEvent(ctypes.Structure):
    _fields_ = [('message', PmMessage), ('timestamp', PmTimestamp)]