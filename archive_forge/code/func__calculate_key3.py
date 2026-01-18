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
def _calculate_key3(password: bytes, cycles: int, salt: bytes, digest: str) -> bytes:
    """Calculate 7zip AES encryption key.
    Concat values in order to reduce number of calls of Hash.update()."""
    assert cycles <= 63
    if cycles == 63:
        ba = bytearray(salt + password + bytes(32))
        key: bytes = bytes(ba[:32])
    else:
        cat_cycle = 6
        if cycles > cat_cycle:
            rounds = 1 << cat_cycle
            stages = 1 << cycles - cat_cycle
        else:
            rounds = 1 << cycles
            stages = 1 << 0
        m = _get_hash(digest)
        saltpassword = salt + password
        s = 0
        if platform.python_implementation() == 'PyPy':
            for _ in range(stages):
                m.update(memoryview(b''.join([saltpassword + (s + i).to_bytes(8, byteorder='little', signed=False) for i in range(rounds)])))
                s += rounds
        else:
            for _ in range(stages):
                m.update(b''.join([saltpassword + (s + i).to_bytes(8, byteorder='little', signed=False) for i in range(rounds)]))
                s += rounds
        key = m.digest()[:32]
    return key