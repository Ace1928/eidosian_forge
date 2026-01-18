import re
import weakref
from ctypes import *
from io import open, BytesIO
import pyglet
from pyglet.gl import *
from pyglet.gl import gl_info
from pyglet.util import asbytes
from .codecs import ImageEncodeException, ImageDecodeException
from .codecs import registry as _codec_registry
from .codecs import add_default_codecs as _add_default_codecs
from .animation import Animation, AnimationFrame
from .buffer import *
from . import atlas
class ColorBufferImage(BufferImage):
    """A color framebuffer.

    This class is used to wrap the primary color buffer (i.e., the back
    buffer)
    """
    gl_format = GL_RGBA
    format = 'RGBA'

    def get_texture(self, rectangle=False):
        texture = Texture.create(self.width, self.height, GL_TEXTURE_2D, GL_RGBA, blank_data=False)
        self.blit_to_texture(texture.target, texture.level, self.anchor_x, self.anchor_y, 0)
        return texture

    def blit_to_texture(self, target, level, x, y, z):
        glReadBuffer(self.gl_buffer)
        glCopyTexSubImage2D(target, level, x - self.anchor_x, y - self.anchor_y, self.x, self.y, self.width, self.height)