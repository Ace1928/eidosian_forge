import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
class MXNetError(RuntimeError):
    """Default error thrown by MXNet functions.

    MXNetError will be raised if you do not give any error type specification,
    """