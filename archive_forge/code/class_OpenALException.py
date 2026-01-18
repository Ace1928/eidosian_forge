import ctypes
import weakref
from collections import namedtuple
from . import lib_openal as al
from . import lib_alc as alc
from pyglet.util import debug_print
from pyglet.media.exceptions import MediaException
class OpenALException(MediaException):

    def __init__(self, message=None, error_code=None, error_string=None):
        self.message = message
        self.error_code = error_code
        self.error_string = error_string

    def __str__(self):
        if self.error_code is None:
            return f'OpenAL Exception: {self.message}'
        else:
            return f'OpenAL Exception [{self.error_code}: {self.error_string}]: {self.message}'