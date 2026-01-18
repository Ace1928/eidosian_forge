from ctypes import *
from .base import FontException
import pyglet.lib
def FT_LOAD_TARGET_(x):
    return (x & 15) << 16