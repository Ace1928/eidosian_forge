import atexit
import struct
import warnings
import pyglet
from . import com
from . import constants
from .types import *
def _uninitialize():
    try:
        _ole32.CoUninitialize()
    except OSError:
        pass