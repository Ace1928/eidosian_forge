import unicodedata
from pyglet.gl import *
from pyglet import image
class FontException(Exception):
    """Generic exception related to errors from the font module.  Typically
    these relate to invalid font data."""
    pass