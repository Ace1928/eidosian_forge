import os
import threading
import subprocess
import sys
import time
import signal
import shlex
from .spawnbase import SpawnBase, PY3
from .exceptions import EOF
from .utils import string_types
def _read_incoming(self):
    """Run in a thread to move output from a pipe to a queue."""
    fileno = self.proc.stdout.fileno()
    while 1:
        buf = b''
        try:
            buf = os.read(fileno, 1024)
        except OSError as e:
            self._log(e, 'read')
        if not buf:
            self._read_queue.put(None)
            return
        self._read_queue.put(buf)