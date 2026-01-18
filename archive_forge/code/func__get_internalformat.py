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
@staticmethod
def _get_internalformat(fmt):
    if fmt == 'R':
        return GL_RED
    elif fmt == 'RG':
        return GL_RG
    elif fmt == 'RGB':
        return GL_RGB
    elif fmt == 'RGBA':
        return GL_RGBA
    elif fmt == 'D':
        return GL_DEPTH_COMPONENT
    elif fmt == 'DS':
        return GL_DEPTH_STENCIL
    elif fmt == 'L':
        return GL_LUMINANCE
    elif fmt == 'A':
        return GL_ALPHA
    return GL_RGBA