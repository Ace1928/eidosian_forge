import sys
import _winapi
import itertools
import msvcrt
import os
import subprocess
import tempfile
import warnings
class PipeHandle:
    """Wrapper for an overlapped pipe handle which is vaguely file-object like.

    The IOCP event loop can use these instead of socket objects.
    """

    def __init__(self, handle):
        self._handle = handle

    def __repr__(self):
        if self._handle is not None:
            handle = f'handle={self._handle!r}'
        else:
            handle = 'closed'
        return f'<{self.__class__.__name__} {handle}>'

    @property
    def handle(self):
        return self._handle

    def fileno(self):
        if self._handle is None:
            raise ValueError('I/O operation on closed pipe')
        return self._handle

    def close(self, *, CloseHandle=_winapi.CloseHandle):
        if self._handle is not None:
            CloseHandle(self._handle)
            self._handle = None

    def __del__(self, _warn=warnings.warn):
        if self._handle is not None:
            _warn(f'unclosed {self!r}', ResourceWarning, source=self)
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, t, v, tb):
        self.close()