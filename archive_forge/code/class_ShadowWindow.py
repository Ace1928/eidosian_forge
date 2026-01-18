import pyglet as _pyglet
from pyglet.gl.gl import *
from pyglet.gl.lib import GLException
from pyglet.gl import gl_info
from pyglet.gl.gl_compat import GL_LUMINANCE, GL_INTENSITY
from pyglet import compat_platform
from .base import ObjectSpace, CanvasConfig, Context
import sys as _sys
class ShadowWindow(Window):

    def __init__(self):
        super().__init__(width=1, height=1, visible=False)

    def _create_projection(self):
        """Shadow window does not need a projection."""
        pass