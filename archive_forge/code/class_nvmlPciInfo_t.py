from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class nvmlPciInfo_t(_PrintableStructure):
    _fields_ = [('busIdLegacy', c_char * NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE), ('domain', c_uint), ('bus', c_uint), ('device', c_uint), ('pciDeviceId', c_uint), ('pciSubSystemId', c_uint), ('busId', c_char * NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE)]
    _fmt_ = {'domain': '0x%08X', 'bus': '0x%02X', 'device': '0x%02X', 'pciDeviceId': '0x%08X', 'pciSubSystemId': '0x%08X'}