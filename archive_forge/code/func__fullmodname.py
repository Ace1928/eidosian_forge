import io
import linecache
import os
import sys
import sysconfig
import token
import tokenize
import inspect
import gc
import dis
import pickle
from time import monotonic as _time
import threading
def _fullmodname(path):
    """Return a plausible module name for the path."""
    comparepath = os.path.normcase(path)
    longest = ''
    for dir in sys.path:
        dir = os.path.normcase(dir)
        if comparepath.startswith(dir) and comparepath[len(dir)] == os.sep:
            if len(dir) > len(longest):
                longest = dir
    if longest:
        base = path[len(longest) + 1:]
    else:
        base = path
    drive, base = os.path.splitdrive(base)
    base = base.replace(os.sep, '.')
    if os.altsep:
        base = base.replace(os.altsep, '.')
    filename, ext = os.path.splitext(base)
    return filename.lstrip('.')