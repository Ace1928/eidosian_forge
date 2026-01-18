import os
import re
import sys
from functools import partial, partialmethod, wraps
from inspect import signature
from unicodedata import east_asian_width
from warnings import warn
from weakref import proxy
def _screen_shape_linux(fp):
    try:
        from array import array
        from fcntl import ioctl
        from termios import TIOCGWINSZ
    except ImportError:
        return (None, None)
    else:
        try:
            rows, cols = array('h', ioctl(fp, TIOCGWINSZ, '\x00' * 8))[:2]
            return (cols, rows)
        except Exception:
            try:
                return [int(os.environ[i]) - 1 for i in ('COLUMNS', 'LINES')]
            except (KeyError, ValueError):
                return (None, None)