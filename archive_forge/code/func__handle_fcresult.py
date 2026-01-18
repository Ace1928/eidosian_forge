from collections import OrderedDict
from ctypes import *
import pyglet.lib
from pyglet.util import asbytes, asstr
from pyglet.font.base import FontException
def _handle_fcresult(result):
    if result == FcResultMatch:
        return True
    elif result in (FcResultNoMatch, FcResultTypeMismatch, FcResultNoId):
        return False
    elif result == FcResultOutOfMemory:
        raise FontException('FontConfig ran out of memory.')