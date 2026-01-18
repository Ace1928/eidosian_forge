import select
import socket
import sys
import time
import warnings
import os
from errno import EALREADY, EINPROGRESS, EWOULDBLOCK, ECONNRESET, EINVAL, \
class file_wrapper:

    def __init__(self, fd):
        self.fd = os.dup(fd)

    def __del__(self):
        if self.fd >= 0:
            warnings.warn('unclosed file %r' % self, ResourceWarning, source=self)
        self.close()

    def recv(self, *args):
        return os.read(self.fd, *args)

    def send(self, *args):
        return os.write(self.fd, *args)

    def getsockopt(self, level, optname, buflen=None):
        if level == socket.SOL_SOCKET and optname == socket.SO_ERROR and (not buflen):
            return 0
        raise NotImplementedError('Only asyncore specific behaviour implemented.')
    read = recv
    write = send

    def close(self):
        if self.fd < 0:
            return
        fd = self.fd
        self.fd = -1
        os.close(fd)

    def fileno(self):
        return self.fd