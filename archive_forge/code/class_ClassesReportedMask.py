import xcffib
import struct
import io
from . import xfixes
from . import xproto
class ClassesReportedMask:
    OutOfProximity = 1 << 7
    DeviceModeAbsolute = 1 << 6
    ReportingValuators = 1 << 2
    ReportingButtons = 1 << 1
    ReportingKeys = 1 << 0