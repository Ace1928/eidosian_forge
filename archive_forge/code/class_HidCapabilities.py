import ctypes
from ctypes import wintypes
import platform
from pyu2f import errors
from pyu2f.hid import base
class HidCapabilities(ctypes.Structure):
    _fields_ = [('Usage', ctypes.c_ushort), ('UsagePage', ctypes.c_ushort), ('InputReportByteLength', ctypes.c_ushort), ('OutputReportByteLength', ctypes.c_ushort), ('FeatureReportByteLength', ctypes.c_ushort), ('Reserved', ctypes.c_ushort * 17), ('NotUsed', ctypes.c_ushort * 10)]