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
def _text_encoding():
    if sys.flags.warn_default_encoding:
        f = sys._getframe()
        filename = f.f_code.co_filename
        stacklevel = 2
        while (f := f.f_back):
            if f.f_code.co_filename != filename:
                break
            stacklevel += 1
        warnings.warn("'encoding' argument not specified.", EncodingWarning, stacklevel)
    if sys.flags.utf8_mode:
        return 'utf-8'
    else:
        return locale.getencoding()