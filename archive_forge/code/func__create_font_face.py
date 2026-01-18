import ctypes
import warnings
from collections import namedtuple
from pyglet.util import asbytes, asstr
from pyglet.font import base
from pyglet import image
from pyglet.font.fontconfig import get_fontconfig
from pyglet.font.freetype_lib import *
def _create_font_face(self):
    ft_library = ft_get_library()
    ft_face = FT_Face()
    FT_New_Memory_Face(ft_library, self.font_data, len(self.font_data), 0, byref(ft_face))
    return ft_face