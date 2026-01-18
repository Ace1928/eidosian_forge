from __future__ import absolute_import, division, print_function
import collections
import sys
import time
import datetime
import os
import platform
import re
import functools
from contextlib import contextmanager
def _genericPyAutoGUIChecks(wrappedFunction):
    """
    A decorator that calls failSafeCheck() before the decorated function and
    _handlePause() after it.
    """

    @functools.wraps(wrappedFunction)
    def wrapper(*args, **kwargs):
        failSafeCheck()
        returnVal = wrappedFunction(*args, **kwargs)
        _handlePause(kwargs.get('_pause', True))
        return returnVal
    return wrapper