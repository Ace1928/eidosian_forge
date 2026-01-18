import ctypes
import enum
import os
import platform
import sys
import numpy as np
def _safe_to_run(self):
    """Returns true if there exist no numpy array buffers.

    This means it is safe to run tflite calls that may destroy internally
    allocated memory. This works, because in the wrapper.cc we have made
    the numpy base be the self._interpreter.
    """
    return sys.getrefcount(self._interpreter) == 2