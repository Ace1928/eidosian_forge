from .base import Display, Screen, ScreenMode, Canvas
from pyglet.libs.win32 import _user32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32.context_managers import device_context
class Win32ScreenMode(ScreenMode):

    def __init__(self, screen, mode):
        super(Win32ScreenMode, self).__init__(screen)
        self._mode = mode
        self.width = mode.dmPelsWidth
        self.height = mode.dmPelsHeight
        self.depth = mode.dmBitsPerPel
        self.rate = mode.dmDisplayFrequency
        self.scaling = mode.dmDisplayFixedOutput

    def __repr__(self):
        return f'{self.__class__.__name__}(width={self.width!r}, height={self.height!r}, depth={self.depth!r}, rate={self.rate}, scaling={self.scaling})'