import os
import signal
import socket
import sys
import threading
from . import process
from .context import reduction
from . import util
def _serve(self):
    if hasattr(signal, 'pthread_sigmask'):
        signal.pthread_sigmask(signal.SIG_BLOCK, signal.valid_signals())
    while 1:
        try:
            with self._listener.accept() as conn:
                msg = conn.recv()
                if msg is None:
                    break
                key, destination_pid = msg
                send, close = self._cache.pop(key)
                try:
                    send(conn, destination_pid)
                finally:
                    close()
        except:
            if not util.is_exiting():
                sys.excepthook(*sys.exc_info())