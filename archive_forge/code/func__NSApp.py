import ctypes
import ctypes.util
from threading import Event
def _NSApp():
    """Return the global NSApplication instance (NSApp)"""
    objc.objc_msgSend.argtypes = [void_p, void_p]
    return msg(C('NSApplication'), n('sharedApplication'))