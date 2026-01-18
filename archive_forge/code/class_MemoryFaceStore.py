import ctypes
import warnings
from collections import namedtuple
from pyglet.util import asbytes, asstr
from pyglet.font import base
from pyglet import image
from pyglet.font.fontconfig import get_fontconfig
from pyglet.font.freetype_lib import *
class MemoryFaceStore:

    def __init__(self):
        self._dict = {}

    def add(self, face):
        self._dict[face.name.lower(), face.bold, face.italic] = face

    def contains(self, name):
        lname = name and name.lower() or ''
        return len([name for name, _, _ in self._dict.keys() if name == lname]) > 0

    def get(self, name, bold, italic):
        lname = name and name.lower() or ''
        return self._dict.get((lname, bold, italic), None)