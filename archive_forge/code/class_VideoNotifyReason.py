import xcffib
import struct
import io
from . import xproto
from . import shm
class VideoNotifyReason:
    Started = 0
    Stopped = 1
    Busy = 2
    Preempted = 3
    HardError = 4