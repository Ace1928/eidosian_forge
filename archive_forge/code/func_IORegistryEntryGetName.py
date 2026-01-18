from __future__ import absolute_import
import ctypes
from serial.tools import list_ports_common
def IORegistryEntryGetName(device):
    devicename = ctypes.create_string_buffer(io_name_size)
    res = iokit.IORegistryEntryGetName(device, ctypes.byref(devicename))
    if res != KERN_SUCCESS:
        return None
    return devicename.value.decode('utf-8')