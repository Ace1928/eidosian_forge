import errno
import os
import socket
import ssl
import stat
import sys
import time
from gunicorn import util
class BaseSocket(object):

    def __init__(self, address, conf, log, fd=None):
        self.log = log
        self.conf = conf
        self.cfg_addr = address
        if fd is None:
            sock = socket.socket(self.FAMILY, socket.SOCK_STREAM)
            bound = False
        else:
            sock = socket.fromfd(fd, self.FAMILY, socket.SOCK_STREAM)
            os.close(fd)
            bound = True
        self.sock = self.set_options(sock, bound=bound)

    def __str__(self):
        return '<socket %d>' % self.sock.fileno()

    def __getattr__(self, name):
        return getattr(self.sock, name)

    def set_options(self, sock, bound=False):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if self.conf.reuse_port and hasattr(socket, 'SO_REUSEPORT'):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except socket.error as err:
                if err.errno not in (errno.ENOPROTOOPT, errno.EINVAL):
                    raise
        if not bound:
            self.bind(sock)
        sock.setblocking(0)
        if hasattr(sock, 'set_inheritable'):
            sock.set_inheritable(True)
        sock.listen(self.conf.backlog)
        return sock

    def bind(self, sock):
        sock.bind(self.cfg_addr)

    def close(self):
        if self.sock is None:
            return
        try:
            self.sock.close()
        except socket.error as e:
            self.log.info('Error while closing socket %s', str(e))
        self.sock = None