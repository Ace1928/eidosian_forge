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
class TextureSequence(AbstractImageSequence):
    """Interface for a sequence of textures.

    Typical implementations store multiple :py:class:`~pyglet.image.TextureRegion` s within one
    :py:class:`~pyglet.image.Texture` so as to minimise state changes.
    """

    def get_texture_sequence(self):
        return self