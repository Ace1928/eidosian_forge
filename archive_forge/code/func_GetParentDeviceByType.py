from __future__ import absolute_import
import ctypes
from serial.tools import list_ports_common
def GetParentDeviceByType(device, parent_type):
    """ Find the first parent of a device that implements the parent_type
        @param IOService Service to inspect
        @return Pointer to the parent type, or None if it was not found.
    """
    parent_type = parent_type.encode('utf-8')
    while IOObjectGetClass(device) != parent_type:
        parent = ctypes.c_void_p()
        response = iokit.IORegistryEntryGetParentEntry(device, 'IOService'.encode('utf-8'), ctypes.byref(parent))
        if response != KERN_SUCCESS:
            return None
        device = parent
    return device