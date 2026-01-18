from __future__ import absolute_import
import ctypes
from serial.tools import list_ports_common
def GetIOServicesByType(service_type):
    """
    returns iterator over specified service_type
    """
    serial_port_iterator = ctypes.c_void_p()
    iokit.IOServiceGetMatchingServices(kIOMasterPortDefault, iokit.IOServiceMatching(service_type.encode('utf-8')), ctypes.byref(serial_port_iterator))
    services = []
    while iokit.IOIteratorIsValid(serial_port_iterator):
        service = iokit.IOIteratorNext(serial_port_iterator)
        if not service:
            break
        services.append(service)
    iokit.IOObjectRelease(serial_port_iterator)
    return services