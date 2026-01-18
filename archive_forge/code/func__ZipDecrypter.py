import binascii
import importlib.util
import io
import itertools
import os
import posixpath
import shutil
import stat
import struct
import sys
import threading
import time
import contextlib
import pathlib
def _ZipDecrypter(pwd):
    key0 = 305419896
    key1 = 591751049
    key2 = 878082192
    global _crctable
    if _crctable is None:
        _crctable = list(map(_gen_crc, range(256)))
    crctable = _crctable

    def crc32(ch, crc):
        """Compute the CRC32 primitive on one byte."""
        return crc >> 8 ^ crctable[(crc ^ ch) & 255]

    def update_keys(c):
        nonlocal key0, key1, key2
        key0 = crc32(c, key0)
        key1 = key1 + (key0 & 255) & 4294967295
        key1 = key1 * 134775813 + 1 & 4294967295
        key2 = crc32(key1 >> 24, key2)
    for p in pwd:
        update_keys(p)

    def decrypter(data):
        """Decrypt a bytes object."""
        result = bytearray()
        append = result.append
        for c in data:
            k = key2 | 2
            c ^= k * (k ^ 1) >> 8 & 255
            update_keys(c)
            append(c)
        return bytes(result)
    return decrypter