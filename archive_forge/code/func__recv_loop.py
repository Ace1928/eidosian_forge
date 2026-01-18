import errno
import os
import socket
import sys
import time
import warnings
import eventlet
from eventlet.hubs import trampoline, notify_opened, IOClosed
from eventlet.support import get_errno
def _recv_loop(self, recv_meth, empty_val, *args):
    if self.act_non_blocking:
        return recv_meth(*args)
    while True:
        try:
            if not args[0]:
                self._read_trampoline()
            return recv_meth(*args)
        except OSError as e:
            if get_errno(e) in SOCKET_BLOCKING:
                pass
            elif get_errno(e) in SOCKET_CLOSED:
                return empty_val
            else:
                raise
        try:
            self._read_trampoline()
        except IOClosed as e:
            raise EOFError()