import ctypes
import warnings
from collections import namedtuple
from pyglet.util import asbytes, asstr
from pyglet.font import base
from pyglet import image
from pyglet.font.fontconfig import get_fontconfig
from pyglet.font.freetype_lib import *
def _copy_font_data(self, data):
    self.font_data = (FT_Byte * len(data))()
    ctypes.memmove(self.font_data, data, len(data))