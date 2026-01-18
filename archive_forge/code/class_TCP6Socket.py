import errno
import os
import socket
import ssl
import stat
import sys
import time
from gunicorn import util
class TCP6Socket(TCPSocket):
    FAMILY = socket.AF_INET6

    def __str__(self):
        host, port, _, _ = self.sock.getsockname()
        return 'http://[%s]:%d' % (host, port)