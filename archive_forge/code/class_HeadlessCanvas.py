import pyglet
import warnings
from .base import Display, Screen, ScreenMode, Canvas
from ctypes import *
from pyglet.libs.egl import egl
from pyglet.libs.egl import eglext
class HeadlessCanvas(Canvas):

    def __init__(self, display, egl_surface):
        super().__init__(display)
        self.egl_surface = egl_surface