import os
from .dependencies import ctypes
def _attempt_ctypes_cdll(name):
    """Load a CDLL library, returning bool indicating success"""
    try:
        dll = ctypes.CDLL(name)
        return True
    except OSError:
        return False