from __future__ import unicode_literals
from ctypes import windll, byref, ArgumentError, c_char, c_long, c_ulong, c_uint, pointer
from ctypes.wintypes import DWORD
from prompt_toolkit.renderer import Output
from prompt_toolkit.styles import ANSI_COLOR_NAMES
from prompt_toolkit.win32_types import CONSOLE_SCREEN_BUFFER_INFO, STD_OUTPUT_HANDLE, STD_INPUT_HANDLE, COORD, SMALL_RECT
import os
import six
def _winapi(self, func, *a, **kw):
    """
        Flush and call win API function.
        """
    self.flush()
    if _DEBUG_RENDER_OUTPUT:
        self.LOG.write(('%r' % func.__name__).encode('utf-8') + b'\n')
        self.LOG.write(b'     ' + ', '.join(['%r' % i for i in a]).encode('utf-8') + b'\n')
        self.LOG.write(b'     ' + ', '.join(['%r' % type(i) for i in a]).encode('utf-8') + b'\n')
        self.LOG.flush()
    try:
        return func(*a, **kw)
    except ArgumentError as e:
        if _DEBUG_RENDER_OUTPUT:
            self.LOG.write(('    Error in %r %r %s\n' % (func.__name__, e, e)).encode('utf-8'))