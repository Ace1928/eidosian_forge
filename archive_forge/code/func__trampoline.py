import errno
import os
import socket
import sys
import time
import warnings
import eventlet
from eventlet.hubs import trampoline, notify_opened, IOClosed
from eventlet.support import get_errno
def _trampoline(self, fd, read=False, write=False, timeout=None, timeout_exc=None):
    """ We need to trampoline via the event hub.
            We catch any signal back from the hub indicating that the operation we
            were waiting on was associated with a filehandle that's since been
            invalidated.
        """
    if self._closed:
        raise IOClosed()
    try:
        return trampoline(fd, read=read, write=write, timeout=timeout, timeout_exc=timeout_exc, mark_as_closed=self._mark_as_closed)
    except IOClosed:
        self._mark_as_closed()
        raise