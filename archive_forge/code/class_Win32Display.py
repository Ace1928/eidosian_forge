from .base import Display, Screen, ScreenMode, Canvas
from pyglet.libs.win32 import _user32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32.context_managers import device_context
class Win32Display(Display):

    def get_screens(self):
        screens = []

        def enum_proc(hMonitor, hdcMonitor, lprcMonitor, dwData):
            r = lprcMonitor.contents
            width = r.right - r.left
            height = r.bottom - r.top
            screens.append(Win32Screen(self, hMonitor, r.left, r.top, width, height))
            return True
        enum_proc_ptr = MONITORENUMPROC(enum_proc)
        _user32.EnumDisplayMonitors(None, None, enum_proc_ptr, 0)
        return screens