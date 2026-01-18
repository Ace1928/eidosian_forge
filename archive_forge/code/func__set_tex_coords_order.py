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
def _set_tex_coords_order(self, bl, br, tr, tl):
    tex_coords = (self.tex_coords[:3], self.tex_coords[3:6], self.tex_coords[6:9], self.tex_coords[9:])
    self.tex_coords = tex_coords[bl] + tex_coords[br] + tex_coords[tr] + tex_coords[tl]
    order = self.tex_coords_order
    self.tex_coords_order = (order[bl], order[br], order[tr], order[tl])