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
class DepthBufferImage(BufferImage):
    """The depth buffer.
    """
    gl_format = GL_DEPTH_COMPONENT
    format = 'R'

    def get_texture(self, rectangle=False):
        image_data = self.get_image_data()
        return image_data.get_texture(rectangle)

    def blit_to_texture(self, target, level, x, y, z):
        glReadBuffer(self.gl_buffer)
        glCopyTexSubImage2D(target, level, x - self.anchor_x, y - self.anchor_y, self.x, self.y, self.width, self.height)