import pyglet as _pyglet
from pyglet.gl.gl import *
from pyglet.gl.lib import GLException
from pyglet.gl import gl_info
from pyglet.gl.gl_compat import GL_LUMINANCE, GL_INTENSITY
from pyglet import compat_platform
from .base import ObjectSpace, CanvasConfig, Context
import sys as _sys
def _debug_texture_alloc(texture, size):
    global _debug_texture_total
    _debug_texture_sizes[texture] = size
    _debug_texture_total += size
    print(f'{_debug_texture_total} (+{size})')