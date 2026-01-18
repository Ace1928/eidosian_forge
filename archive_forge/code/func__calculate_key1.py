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
def _calculate_key1(password: bytes, cycles: int, salt: bytes, digest: str) -> bytes:
    """Calculate 7zip AES encryption key. Base implementation."""
    assert cycles <= 63
    if cycles == 63:
        ba = bytearray(salt + password + bytes(32))
        key: bytes = bytes(ba[:32])
    else:
        rounds = 1 << cycles
        m = _get_hash(digest)
        for round in range(rounds):
            m.update(salt + password + round.to_bytes(8, byteorder='little', signed=False))
        key = m.digest()[:32]
    return key