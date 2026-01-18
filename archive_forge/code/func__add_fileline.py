import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
def _add_fileline(obj):
    """Add fileinto to a object.
        """
    if obj.__doc__ is None or 'From:' in obj.__doc__:
        return
    fname = inspect.getsourcefile(obj)
    if fname is None:
        return
    try:
        line = inspect.getsourcelines(obj)[-1]
    except IOError:
        return
    obj.__doc__ += '\n\nFrom:%s:%d' % (fname, line)