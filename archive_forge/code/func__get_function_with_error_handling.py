from ctypes import *
from .base import FontException
import pyglet.lib
def _get_function_with_error_handling(name, argtypes, rtype):
    func = _get_function(name, argtypes, rtype)

    def _error_handling(*args, **kwargs):
        err = func(*args, **kwargs)
        FreeTypeError.check_and_raise_on_error(err)
    return _error_handling