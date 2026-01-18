import ctypes
import ctypes.util
from threading import Event
def _input_callback(fdref, flags, info):
    """Callback to fire when there's input to be read"""
    CFFileDescriptorInvalidate(fdref)
    CFRelease(fdref)
    NSApp = _NSApp()
    objc.objc_msgSend.argtypes = [void_p, void_p, void_p]
    msg(NSApp, n('stop:'), NSApp)
    _wake(NSApp)