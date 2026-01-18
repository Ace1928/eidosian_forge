import pyglet
import warnings
from .base import Display, Screen, ScreenMode, Canvas
from ctypes import *
from pyglet.libs.egl import egl
from pyglet.libs.egl import eglext
class HeadlessScreen(Screen):

    def __init__(self, display, x, y, width, height):
        super().__init__(display, x, y, width, height)

    def get_matching_configs(self, template):
        canvas = HeadlessCanvas(self.display, None)
        configs = template.match(canvas)
        for config in configs:
            config.screen = self
        return configs

    def get_modes(self):
        pass

    def get_mode(self):
        pass

    def set_mode(self, mode):
        pass

    def restore_mode(self):
        pass