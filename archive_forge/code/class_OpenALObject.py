import ctypes
import weakref
from collections import namedtuple
from . import lib_openal as al
from . import lib_alc as alc
from pyglet.util import debug_print
from pyglet.media.exceptions import MediaException
class OpenALObject:
    """Base class for OpenAL objects."""

    @classmethod
    def _check_error(cls, message=None):
        """Check whether there is an OpenAL error and raise exception if present."""
        error_code = al.alGetError()
        if error_code != 0:
            error_string = al.alGetString(error_code)
            error_string = ctypes.cast(error_string, ctypes.c_char_p)
            raise OpenALException(message=message, error_code=error_code, error_string=str(error_string.value))