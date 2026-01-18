import base64
import codecs
import collections
import errno
from random import Random
from socket import error as SocketError
import string
import struct
import sys
import time
import zlib
from eventlet import semaphore
from eventlet import wsgi
from eventlet.green import socket
from eventlet.support import get_errno
def _send_closing_frame(self, ignore_send_errors=False, close_data=None):
    if self.version in (8, 13) and (not self.websocket_closed):
        if close_data is not None:
            status, msg = close_data
            if isinstance(msg, str):
                msg = msg.encode('utf-8')
            data = struct.pack('!H', status) + msg
        else:
            data = ''
        try:
            self.send(data, control_code=8)
        except OSError:
            if not ignore_send_errors:
                raise
        self.websocket_closed = True