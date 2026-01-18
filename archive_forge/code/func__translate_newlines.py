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
def _translate_newlines(self, data, encoding, errors):
    data = data.decode(encoding, errors)
    return data.replace('\r\n', '\n').replace('\r', '\n')