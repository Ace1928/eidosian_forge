import _signal
from _signal import *
from enum import IntEnum as _IntEnum
def _wraps(wrapped):

    def decorator(wrapper):
        wrapper.__doc__ = wrapped.__doc__
        return wrapper
    return decorator