import ctypes
import enum
import os
import platform
import sys
import numpy as np
def _get_native_delegate_pointer(self):
    """Returns the native TfLiteDelegate pointer.

    It is not safe to copy this pointer because it needs to be freed.

    Returns:
      TfLiteDelegate *
    """
    return self._delegate_ptr