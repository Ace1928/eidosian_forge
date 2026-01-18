import builtins
import errno
import io
import locale
import os
import time
import signal
import sys
import threading
import warnings
import contextlib
from time import monotonic as _time
import types
def _make_inheritable(self, handle):
    """Return a duplicate of handle, which is inheritable"""
    h = _winapi.DuplicateHandle(_winapi.GetCurrentProcess(), handle, _winapi.GetCurrentProcess(), 0, 1, _winapi.DUPLICATE_SAME_ACCESS)
    return Handle(h)