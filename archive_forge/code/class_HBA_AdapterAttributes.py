import ctypes
import os_win.conf
from os_win.utils.winapi import wintypes
class HBA_AdapterAttributes(ctypes.Structure):
    _fields_ = [('Manufacturer', wintypes.CHAR * 64), ('SerialNumber', wintypes.CHAR * 64), ('Model', wintypes.CHAR * 256), ('ModelDescription', wintypes.CHAR * 256), ('NodeWWN', HBA_WWN), ('NodeSymbolicName', wintypes.CHAR * 256), ('HardwareVersion', wintypes.CHAR * 256), ('DriverVersion', wintypes.CHAR * 256), ('OptionROMVersion', wintypes.CHAR * 256), ('FirmwareVersion', wintypes.CHAR * 256), ('VendorSpecificID', ctypes.c_uint32), ('NumberOfPorts', ctypes.c_uint32), ('DriverName', wintypes.CHAR * 256)]