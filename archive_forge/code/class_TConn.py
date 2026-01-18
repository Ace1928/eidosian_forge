from concurrent import futures
import errno
import os
import selectors
import socket
import ssl
import sys
import time
from collections import deque
from datetime import datetime
from functools import partial
from threading import RLock
from . import base
from .. import http
from .. import util
from .. import sock
from ..http import wsgi
class TConn(object):

    def __init__(self, cfg, sock, client, server):
        self.cfg = cfg
        self.sock = sock
        self.client = client
        self.server = server
        self.timeout = None
        self.parser = None
        self.initialized = False
        self.sock.setblocking(False)

    def init(self):
        self.initialized = True
        self.sock.setblocking(True)
        if self.parser is None:
            if self.cfg.is_ssl:
                self.sock = sock.ssl_wrap_socket(self.sock, self.cfg)
            self.parser = http.RequestParser(self.cfg, self.sock, self.client)

    def set_timeout(self):
        self.timeout = time.time() + self.cfg.keepalive

    def close(self):
        util.close(self.sock)