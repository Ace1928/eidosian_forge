import ctypes
from ctypes import *
from pyglet import app
from pyglet.app.xlib import XlibSelectDevice
from .base import Display, Screen, ScreenMode, Canvas
from . import xlib_vidmoderestore
from pyglet.libs.x11 import xlib
class XlibScreen(Screen):
    _initial_mode = None

    def __init__(self, display, x, y, width, height, xinerama):
        super(XlibScreen, self).__init__(display, x, y, width, height)
        self._xinerama = xinerama

    def get_matching_configs(self, template):
        canvas = XlibCanvas(self.display, None)
        configs = template.match(canvas)
        for config in configs:
            config.screen = self
        return configs

    def get_modes(self):
        if not _have_xf86vmode:
            return []
        if self._xinerama:
            return []
        count = ctypes.c_int()
        info_array = ctypes.POINTER(ctypes.POINTER(xf86vmode.XF86VidModeModeInfo))()
        xf86vmode.XF86VidModeGetAllModeLines(self.display._display, self.display.x_screen, count, info_array)
        modes = []
        for i in range(count.value):
            info = xf86vmode.XF86VidModeModeInfo()
            ctypes.memmove(ctypes.byref(info), ctypes.byref(info_array.contents[i]), ctypes.sizeof(info))
            modes.append(XlibScreenMode(self, info))
            if info.privsize:
                xlib.XFree(info.private)
        xlib.XFree(info_array)
        return modes

    def get_mode(self):
        modes = self.get_modes()
        if modes:
            return modes[0]
        return None

    def set_mode(self, mode):
        assert mode.screen is self
        if not self._initial_mode:
            self._initial_mode = self.get_mode()
            xlib_vidmoderestore.set_initial_mode(self._initial_mode)
        xf86vmode.XF86VidModeSwitchToMode(self.display._display, self.display.x_screen, mode.info)
        xlib.XFlush(self.display._display)
        xf86vmode.XF86VidModeSetViewPort(self.display._display, self.display.x_screen, 0, 0)
        xlib.XFlush(self.display._display)
        self.width = mode.width
        self.height = mode.height

    def restore_mode(self):
        if self._initial_mode:
            self.set_mode(self._initial_mode)

    def __repr__(self):
        return f'{self.__class__.__name__}(display={self.display!r}, x={self.x}, y={self.y}, width={self.width}, height={self.height}, xinerama={self._xinerama})'