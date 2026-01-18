import ctypes
import warnings
from collections import namedtuple
from pyglet.util import asbytes, asstr
from pyglet.font import base
from pyglet import image
from pyglet.font.fontconfig import get_fontconfig
from pyglet.font.freetype_lib import *
def _load_font_face_from_system(self):
    match = get_fontconfig().find_font(self._name, self.size, self.bold, self.italic)
    if not match:
        raise base.FontException(f"Could not match font '{self._name}'")
    self.filename = match.file
    self.face = FreeTypeFace.from_fontconfig(match)