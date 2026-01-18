import datetime
import struct
from six.moves import winreg
from six import text_type
from ._common import tzrangebase
def _settzkeyname():
    handle = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
    try:
        winreg.OpenKey(handle, TZKEYNAMENT).Close()
        TZKEYNAME = TZKEYNAMENT
    except WindowsError:
        TZKEYNAME = TZKEYNAME9X
    handle.Close()
    return TZKEYNAME