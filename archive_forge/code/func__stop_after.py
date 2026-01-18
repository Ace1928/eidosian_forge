import ctypes
import ctypes.util
from threading import Event
def _stop_after(delay):
    """Register callback to stop eventloop after a delay"""
    timer = CFRunLoopTimerCreate(None, CFAbsoluteTimeGetCurrent() + delay, 0, 0, 0, _c_stop_callback, None)
    CFRunLoopAddTimer(CFRunLoopGetMain(), timer, kCFRunLoopCommonModes)