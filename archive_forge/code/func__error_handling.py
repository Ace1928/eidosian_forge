from ctypes import *
from .base import FontException
import pyglet.lib
def _error_handling(*args, **kwargs):
    err = func(*args, **kwargs)
    FreeTypeError.check_and_raise_on_error(err)