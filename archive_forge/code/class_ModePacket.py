import ctypes
import os
import signal
import struct
import threading
from pyglet.libs.x11 import xlib
from pyglet.util import asbytes
class ModePacket:
    format = '256siHHI'
    size = struct.calcsize(format)

    def __init__(self, display, screen, width, height, rate):
        self.display = display
        self.screen = screen
        self.width = width
        self.height = height
        self.rate = rate

    def encode(self):
        return struct.pack(self.format, self.display, self.screen, self.width, self.height, self.rate)

    @classmethod
    def decode(cls, data):
        display, screen, width, height, rate = struct.unpack(cls.format, data)
        return cls(display.strip(asbytes('\x00')), screen, width, height, rate)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.display}, {self.screen}, {self.width}, {self.height}, {self.rate})'

    def set(self):
        display = xlib.XOpenDisplay(self.display)
        modes, n_modes = get_modes_array(display, self.screen)
        mode = get_matching_mode(modes, n_modes, self.width, self.height, self.rate)
        if mode is not None:
            xf86vmode.XF86VidModeSwitchToMode(display, self.screen, mode)
        free_modes_array(modes, n_modes)
        xlib.XCloseDisplay(display)