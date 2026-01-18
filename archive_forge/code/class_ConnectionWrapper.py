import io
import os
import sys
import socket
import struct
import time
import tempfile
import itertools
from . import util
from . import AuthenticationError, BufferTooShort
from .context import reduction
class ConnectionWrapper(object):

    def __init__(self, conn, dumps, loads):
        self._conn = conn
        self._dumps = dumps
        self._loads = loads
        for attr in ('fileno', 'close', 'poll', 'recv_bytes', 'send_bytes'):
            obj = getattr(conn, attr)
            setattr(self, attr, obj)

    def send(self, obj):
        s = self._dumps(obj)
        self._conn.send_bytes(s)

    def recv(self):
        s = self._conn.recv_bytes()
        return self._loads(s)