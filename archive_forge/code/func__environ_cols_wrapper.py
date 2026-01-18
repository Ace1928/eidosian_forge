import os
import re
import sys
from functools import partial, partialmethod, wraps
from inspect import signature
from unicodedata import east_asian_width
from warnings import warn
from weakref import proxy
def _environ_cols_wrapper():
    """
    Return a function which returns console width.
    Supported: linux, osx, windows, cygwin.
    """
    warn('Use `_screen_shape_wrapper()(file)[0]` instead of `_environ_cols_wrapper()(file)`', DeprecationWarning, stacklevel=2)
    shape = _screen_shape_wrapper()
    if not shape:
        return None

    @wraps(shape)
    def inner(fp):
        return shape(fp)[0]
    return inner