import ctypes
import hashlib
import os
import pathlib
import platform
import sys
import time as _time
import zlib
from datetime import datetime, timedelta, timezone, tzinfo
from typing import BinaryIO, List, Optional, Union
import py7zr.win32compat
from py7zr import Bad7zFile
from py7zr.win32compat import is_windows_native_python, is_windows_unc_path
class NullIO:
    """pathlib.Path-like IO class of /dev/null"""

    def __init__(self):
        pass

    def write(self, data):
        return len(data)

    def read(self, length=None):
        if length is not None:
            return bytes(length)
        else:
            return b''

    def close(self):
        pass

    def flush(self):
        pass

    def open(self, mode=None):
        return self

    @property
    def parent(self):
        return self

    def mkdir(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass