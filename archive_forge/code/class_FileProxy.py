from code import InteractiveConsole
import errno
import socket
import sys
import eventlet
from eventlet import hubs
from eventlet.support import greenlets, get_errno
class FileProxy:

    def __init__(self, f):
        self.f = f

    def isatty(self):
        return True

    def flush(self):
        pass

    def write(self, data, *a, **kw):
        try:
            self.f.write(data, *a, **kw)
            self.f.flush()
        except OSError as e:
            if get_errno(e) != errno.EPIPE:
                raise

    def readline(self, *a):
        return self.f.readline(*a).replace('\r\n', '\n')

    def __getattr__(self, attr):
        return getattr(self.f, attr)