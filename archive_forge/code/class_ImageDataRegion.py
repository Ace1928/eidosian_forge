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
class ImageDataRegion(ImageData):

    def __init__(self, x, y, width, height, image_data):
        super().__init__(width, height, image_data._current_format, image_data._current_data, image_data._current_pitch)
        self.x = x
        self.y = y

    def __getstate__(self):
        return {'width': self.width, 'height': self.height, '_current_data': self.get_data(self._current_format, self._current_pitch), '_current_format': self._current_format, '_desired_format': self._desired_format, '_current_pitch': self._current_pitch, 'pitch': self.pitch, 'mipmap_images': self.mipmap_images, 'x': self.x, 'y': self.y}

    def get_data(self, fmt=None, pitch=None):
        x1 = len(self._current_format) * self.x
        x2 = len(self._current_format) * (self.x + self.width)
        self._ensure_bytes()
        data = self._convert(self._current_format, abs(self._current_pitch))
        new_pitch = abs(self._current_pitch)
        rows = [data[i:i + new_pitch] for i in range(0, len(data), new_pitch)]
        rows = [row[x1:x2] for row in rows[self.y:self.y + self.height]]
        self._current_data = b''.join(rows)
        self._current_pitch = self.width * len(self._current_format)
        self._current_texture = None
        self.x = 0
        self.y = 0
        fmt = fmt or self._desired_format
        pitch = pitch or self._current_pitch
        return super().get_data(fmt, pitch)

    def set_data(self, fmt, pitch, data):
        self.x = 0
        self.y = 0
        super().set_data(fmt, pitch, data)

    def _apply_region_unpack(self):
        glPixelStorei(GL_UNPACK_SKIP_PIXELS, self.x)
        glPixelStorei(GL_UNPACK_SKIP_ROWS, self.y)

    def _default_region_unpack(self):
        glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0)
        glPixelStorei(GL_UNPACK_SKIP_ROWS, 0)

    def get_region(self, x, y, width, height):
        x += self.x
        y += self.y
        return super().get_region(x, y, width, height)