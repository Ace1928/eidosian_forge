from os.path import join
import sys
from typing import Optional
from kivy import kivy_data_dir
from kivy.logger import Logger
from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import WindowBase
from kivy.input.provider import MotionEventProvider
from kivy.input.motionevent import MotionEvent
from kivy.resources import resource_find
from kivy.utils import platform, deprecated
from kivy.compat import unichr
from collections import deque
def _wnd_proc(self, hwnd, msg, wParam, lParam):
    from kivy.input.providers.wm_common import WM_DPICHANGED, WM_NCCALCSIZE
    from ctypes import windll
    if msg == WM_DPICHANGED:

        def clock_callback(*args):
            if x_dpi != y_dpi:
                raise ValueError('Can only handle DPI that are same for x and y')
            self.window.dpi = x_dpi
        x_dpi = wParam & 65535
        y_dpi = wParam >> 16
        Clock.schedule_once(clock_callback, -1)
    elif Config.getboolean('graphics', 'resizable') and msg == WM_NCCALCSIZE and self.window.custom_titlebar:
        return 0
    return windll.user32.CallWindowProcW(self.old_windProc, hwnd, msg, wParam, lParam)