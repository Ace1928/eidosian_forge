import os
import tempfile
from django.core.files.utils import FileProxyMixin
class TemporaryFile(FileProxyMixin):
    """
        Temporary file object constructor that supports reopening of the
        temporary file in Windows.

        Unlike tempfile.NamedTemporaryFile from the standard library,
        __init__() doesn't support the 'delete', 'buffering', 'encoding', or
        'newline' keyword arguments.
        """

    def __init__(self, mode='w+b', bufsize=-1, suffix='', prefix='', dir=None):
        fd, name = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir)
        self.name = name
        self.file = os.fdopen(fd, mode, bufsize)
        self.close_called = False
    unlink = os.unlink

    def close(self):
        if not self.close_called:
            self.close_called = True
            try:
                self.file.close()
            except OSError:
                pass
            try:
                self.unlink(self.name)
            except OSError:
                pass

    def __del__(self):
        self.close()

    def __enter__(self):
        self.file.__enter__()
        return self

    def __exit__(self, exc, value, tb):
        self.file.__exit__(exc, value, tb)