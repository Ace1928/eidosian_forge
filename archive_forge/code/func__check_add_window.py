import binascii
import os
import socket
import time
import threading
from functools import wraps
from paramiko import util
from paramiko.common import (
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
from paramiko.file import BufferedFile
from paramiko.buffered_pipe import BufferedPipe, PipeTimeout
from paramiko import pipe
from paramiko.util import ClosingContextManager
def _check_add_window(self, n):
    self.lock.acquire()
    try:
        if self.closed or self.eof_received or (not self.active):
            return 0
        if self.ultra_debug:
            self._log(DEBUG, 'addwindow {}'.format(n))
        self.in_window_sofar += n
        if self.in_window_sofar <= self.in_window_threshold:
            return 0
        if self.ultra_debug:
            self._log(DEBUG, 'addwindow send {}'.format(self.in_window_sofar))
        out = self.in_window_sofar
        self.in_window_sofar = 0
        return out
    finally:
        self.lock.release()