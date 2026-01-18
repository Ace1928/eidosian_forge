import ctypes
import os_win.conf
from os_win.utils.winapi import wintypes
class HBA_PortAttributes(ctypes.Structure):
    _fields_ = [('NodeWWN', HBA_WWN), ('PortWWN', HBA_WWN), ('PortFcId', ctypes.c_uint32), ('PortType', HBA_PortType), ('PortState', HBA_PortState), ('PortSupportedClassofService', HBA_COS), ('PortSupportedFc4Types', HBA_FC4Types), ('PortSymbolicName', wintypes.CHAR * 256), ('OSDeviceName', wintypes.CHAR * 256), ('PortSupportedSpeed', HBA_PortSpeed), ('PortSpeed', HBA_PortSpeed), ('PortMaxFrameSize', ctypes.c_uint32), ('FabricName', HBA_WWN), ('NumberOfDiscoveredPorts', ctypes.c_uint32)]