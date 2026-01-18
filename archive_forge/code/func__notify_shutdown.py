import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
def _notify_shutdown():
    """Notify MXNet about a shutdown."""
    check_call(_LIB.MXNotifyShutdown())