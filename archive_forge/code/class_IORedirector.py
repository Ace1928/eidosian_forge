from _pydevd_bundle.pydevd_constants import ForkSafeLock, get_global_debugger
import os
import sys
from contextlib import contextmanager
class IORedirector:
    """
    This class works to wrap a stream (stdout/stderr) with an additional redirect.
    """

    def __init__(self, original, new_redirect, wrap_buffer=False):
        """
        :param stream original:
            The stream to be wrapped (usually stdout/stderr, but could be None).

        :param stream new_redirect:
            Usually IOBuf (below).

        :param bool wrap_buffer:
            Whether to create a buffer attribute (needed to mimick python 3 s
            tdout/stderr which has a buffer to write binary data).
        """
        self._lock = ForkSafeLock(rlock=True)
        self._writing = False
        self._redirect_to = (original, new_redirect)
        if wrap_buffer and hasattr(original, 'buffer'):
            self.buffer = IORedirector(original.buffer, new_redirect.buffer, False)

    def write(self, s):
        with self._lock:
            if self._writing:
                return
            self._writing = True
            try:
                for r in self._redirect_to:
                    if hasattr(r, 'write'):
                        r.write(s)
            finally:
                self._writing = False

    def isatty(self):
        for r in self._redirect_to:
            if hasattr(r, 'isatty'):
                return r.isatty()
        return False

    def flush(self):
        for r in self._redirect_to:
            if hasattr(r, 'flush'):
                r.flush()

    def __getattr__(self, name):
        for r in self._redirect_to:
            if hasattr(r, name):
                return getattr(r, name)
        raise AttributeError(name)