import ctypes
import warnings
from collections import namedtuple
from pyglet.util import asbytes, asstr
from pyglet.font import base
from pyglet import image
from pyglet.font.fontconfig import get_fontconfig
from pyglet.font.freetype_lib import *
def _get_best_name(self):
    self._name = asstr(self.ft_face.contents.family_name)
    self._get_font_family_from_ttf