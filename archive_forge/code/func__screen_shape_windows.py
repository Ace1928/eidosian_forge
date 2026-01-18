import os
import re
import sys
from functools import partial, partialmethod, wraps
from inspect import signature
from unicodedata import east_asian_width
from warnings import warn
from weakref import proxy
def _screen_shape_windows(fp):
    try:
        import struct
        from ctypes import create_string_buffer, windll
        from sys import stdin, stdout
        io_handle = -12
        if fp == stdin:
            io_handle = -10
        elif fp == stdout:
            io_handle = -11
        h = windll.kernel32.GetStdHandle(io_handle)
        csbi = create_string_buffer(22)
        res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
        if res:
            _bufx, _bufy, _curx, _cury, _wattr, left, top, right, bottom, _maxx, _maxy = struct.unpack('hhhhHhhhhhh', csbi.raw)
            return (right - left, bottom - top)
    except Exception:
        pass
    return (None, None)