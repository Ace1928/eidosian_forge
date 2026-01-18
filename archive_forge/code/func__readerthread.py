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
def _readerthread(self, fh, buffer):
    buffer.append(fh.read())
    fh.close()