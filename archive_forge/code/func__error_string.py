from ctypes import (
import ctypes
from ctypes.util import find_library
import logging
import mmap
import os
import sysconfig
from .exception import ArchiveError
def _error_string(archive_p):
    msg = error_string(archive_p)
    if msg is None:
        return
    try:
        return msg.decode('ascii')
    except UnicodeDecodeError:
        return msg