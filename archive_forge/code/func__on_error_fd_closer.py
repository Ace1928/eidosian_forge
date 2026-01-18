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
@contextlib.contextmanager
def _on_error_fd_closer(self):
    """Helper to ensure file descriptors opened in _get_handles are closed"""
    to_close = []
    try:
        yield to_close
    except:
        if hasattr(self, '_devnull'):
            to_close.append(self._devnull)
            del self._devnull
        for fd in to_close:
            try:
                if _mswindows and isinstance(fd, Handle):
                    fd.Close()
                else:
                    os.close(fd)
            except OSError:
                pass
        raise