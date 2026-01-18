import ctypes
import ctypes.util
from threading import Event
def _utf8(s):
    """ensure utf8 bytes"""
    if not isinstance(s, bytes):
        s = s.encode('utf8')
    return s