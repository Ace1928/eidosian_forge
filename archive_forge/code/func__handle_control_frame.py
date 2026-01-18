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
def _handle_control_frame(self, opcode, data):
    if opcode == 8:
        self._remote_close_data = data
        if not data:
            status = 1000
        elif len(data) > 1:
            status = struct.unpack_from('!H', data)[0]
            if not status or status not in VALID_CLOSE_STATUS:
                raise FailedConnectionError(1002, 'Unexpected close status code.')
            try:
                data = self.UTF8Decoder().decode(data[2:], True)
            except (UnicodeDecodeError, ValueError):
                raise FailedConnectionError(1002, 'Close message data should be valid UTF-8.')
        else:
            status = 1002
        self.close(close_data=(status, ''))
        raise ConnectionClosedError()
    elif opcode == 9:
        self.send(data, control_code=10)
    elif opcode == 10:
        pass
    else:
        raise FailedConnectionError(1002, 'Unknown control frame received.')