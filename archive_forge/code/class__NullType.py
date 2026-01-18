import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
class _NullType(object):
    """Placeholder for arguments"""

    def __repr__(self):
        return '_Null'