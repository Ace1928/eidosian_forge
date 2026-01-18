from eventlet.patcher import slurp_properties
import sys
from eventlet import greenio, hubs
from eventlet.greenio import (
from eventlet.hubs import trampoline, IOClosed
from eventlet.support import get_errno, PY33
from contextlib import contextmanager
def _base_recv(self, nbytes, flags, into, buffer_=None):
    if into:
        plain_socket_function = socket.recv_into
    else:
        plain_socket_function = socket.recv
    if self._sslobj:
        if flags != 0:
            raise ValueError('non-zero flags not allowed in calls to %s() on %s' % plain_socket_function.__name__, self.__class__)
        if into:
            read = self.read(nbytes, buffer_)
        else:
            read = self.read(nbytes)
        return read
    else:
        while True:
            try:
                args = [self, nbytes, flags]
                if into:
                    args.insert(1, buffer_)
                return plain_socket_function(*args)
            except orig_socket.error as e:
                if self.act_non_blocking:
                    raise
                erno = get_errno(e)
                if erno in greenio.SOCKET_BLOCKING:
                    try:
                        trampoline(self, read=True, timeout=self.gettimeout(), timeout_exc=timeout_exc('timed out'))
                    except IOClosed:
                        return b''
                elif erno in greenio.SOCKET_CLOSED:
                    return b''
                raise