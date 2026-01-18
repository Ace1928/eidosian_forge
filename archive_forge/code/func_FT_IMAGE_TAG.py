from ctypes import *
from .base import FontException
import pyglet.lib
def FT_IMAGE_TAG(tag):
    return ord(tag[0]) << 24 | ord(tag[1]) << 16 | ord(tag[2]) << 8 | ord(tag[3])