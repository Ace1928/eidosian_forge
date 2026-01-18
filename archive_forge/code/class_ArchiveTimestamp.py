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
class ArchiveTimestamp(int):
    """Windows FILETIME timestamp."""

    def __repr__(self):
        return '%s(%d)' % (type(self).__name__, self)

    def __index__(self):
        return self.__int__()

    def totimestamp(self) -> float:
        """Convert 7z FILETIME to Python timestamp."""
        return self / 10000000.0 + TIMESTAMP_ADJUST

    def as_datetime(self):
        """Convert FILETIME to Python datetime object."""
        return datetime.fromtimestamp(self.totimestamp(), UTC())

    @staticmethod
    def from_datetime(val):
        return ArchiveTimestamp((val - TIMESTAMP_ADJUST) * 10000000.0)

    @staticmethod
    def from_now():
        return ArchiveTimestamp((_time.time() - TIMESTAMP_ADJUST) * 10000000.0)