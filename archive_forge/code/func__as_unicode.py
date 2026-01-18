import os
from .dependencies import ctypes
def _as_unicode(val):
    """Helper function to coerce a string to a unicode() object"""
    if isinstance(val, str):
        return val
    elif val is not None:
        return val.decode()
    return None