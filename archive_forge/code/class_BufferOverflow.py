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
class BufferOverflow(Exception):
    pass