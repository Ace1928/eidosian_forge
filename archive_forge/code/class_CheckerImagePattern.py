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
class CheckerImagePattern(ImagePattern):
    """Create an image with a tileable checker image.
    """

    def __init__(self, color1=(150, 150, 150, 255), color2=(200, 200, 200, 255)):
        """Initialise with the given colors.

        :Parameters:
            `color1` : (int, int, int, int)
                4-tuple of ints in range [0,255] giving RGBA components of
                color to fill with.  This color appears in the top-left and
                bottom-right corners of the image.
            `color2` : (int, int, int, int)
                4-tuple of ints in range [0,255] giving RGBA components of
                color to fill with.  This color appears in the top-right and
                bottom-left corners of the image.

        """
        self.color1 = _color_as_bytes(color1)
        self.color2 = _color_as_bytes(color2)

    def create_image(self, width, height):
        hw = width // 2
        hh = height // 2
        row1 = self.color1 * hw + self.color2 * hw
        row2 = self.color2 * hw + self.color1 * hw
        data = row1 * hh + row2 * hh
        return ImageData(width, height, 'RGBA', data)