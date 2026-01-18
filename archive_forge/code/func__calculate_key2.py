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
def _calculate_key2(password: bytes, cycles: int, salt: bytes, digest: str):
    """Calculate 7zip AES encryption key.
    It uses ctypes and memoryview buffer and zero-copy technology on Python."""
    assert cycles <= 63
    if cycles == 63:
        key: bytes = bytes(bytearray(salt + password + bytes(32))[:32])
    else:
        rounds = 1 << cycles
        m = _get_hash(digest)
        length = len(salt) + len(password)

        class RoundBuf(ctypes.LittleEndianStructure):
            _pack_ = 1
            _fields_ = [('saltpassword', ctypes.c_ubyte * length), ('round', ctypes.c_uint64)]
        buf = RoundBuf()
        for i, c in enumerate(salt + password):
            buf.saltpassword[i] = c
        buf.round = 0
        mv = memoryview(buf)
        while buf.round < rounds:
            m.update(mv)
            buf.round += 1
        key = m.digest()[:32]
    return key