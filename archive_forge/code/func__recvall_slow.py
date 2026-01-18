import base64
import errno
import json
from multiprocessing import connection
from multiprocessing import managers
import socket
import struct
import weakref
from oslo_rootwrap import wrapper
def _recvall_slow(self, size):
    remaining = size
    res = []
    while remaining:
        piece = self._socket.recv(remaining)
        if not piece:
            raise EOFError
        res.append(piece)
        remaining -= len(piece)
    return b''.join(res)