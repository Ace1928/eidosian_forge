import os
import signal
import socket
import sys
import threading
from . import process
from .context import reduction
from . import util
def _afterfork(self):
    for key, (send, close) in self._cache.items():
        close()
    self._cache.clear()
    self._lock._at_fork_reinit()
    if self._listener is not None:
        self._listener.close()
    self._listener = None
    self._address = None
    self._thread = None