import ctypes
from ctypes import wintypes # We can't use ctypes.wintypes, we must import wintypes this way.
from pygetwindow import PyGetWindowException, pointInRect, BaseWindow, Rect, Point, Size
def _raiseWithLastError():
    """A helper function that raises PyGetWindowException using the error
    information from GetLastError() and FormatMessage()."""
    errorCode = ctypes.windll.kernel32.GetLastError()
    raise PyGetWindowException('Error code from Windows: %s - %s' % (errorCode, _formatMessage(errorCode)))