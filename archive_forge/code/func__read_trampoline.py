import errno
import os
import socket
import sys
import time
import warnings
import eventlet
from eventlet.hubs import trampoline, notify_opened, IOClosed
from eventlet.support import get_errno
def _read_trampoline(self):
    self._trampoline(self.fd, read=True, timeout=self.gettimeout(), timeout_exc=socket_timeout('timed out'))