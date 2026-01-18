import inspect
import io
import os
import platform
import warnings
import numpy
import cupy
import cupy_backends
def _dir_or_none(path):
    """Returns None if path does not exist."""
    if os.path.isdir(path):
        return path
    return None