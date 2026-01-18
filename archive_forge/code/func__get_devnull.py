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
def _get_devnull(self):
    if not hasattr(self, '_devnull'):
        self._devnull = os.open(os.devnull, os.O_RDWR)
    return self._devnull