import errno
import os
import socket
import sys
import time
import warnings
import eventlet
from eventlet.hubs import trampoline, notify_opened, IOClosed
from eventlet.support import get_errno
def _send_loop(self, send_method, data, *args):
    if self.act_non_blocking:
        return send_method(data, *args)
    _timeout_exc = socket_timeout('timed out')
    while True:
        try:
            return send_method(data, *args)
        except OSError as e:
            eno = get_errno(e)
            if eno == errno.ENOTCONN or eno not in SOCKET_BLOCKING:
                raise
        try:
            self._trampoline(self.fd, write=True, timeout=self.gettimeout(), timeout_exc=_timeout_exc)
        except IOClosed:
            raise OSError(errno.ECONNRESET, 'Connection closed by another thread')