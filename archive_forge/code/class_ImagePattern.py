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
class ImagePattern:
    """Abstract image creation class."""

    def create_image(self, width, height):
        """Create an image of the given size.

        :Parameters:
            `width` : int
                Width of image to create
            `height` : int
                Height of image to create

        :rtype: AbstractImage
        """
        raise NotImplementedError('abstract')