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
def _verify_driver_supported(self):
    """Assert that the extension required for this image data is
        supported.

        Raises `ImageException` if not.
        """
    if not self._have_extension():
        raise ImageException('%s is required to decode %r' % (self.extension, self))