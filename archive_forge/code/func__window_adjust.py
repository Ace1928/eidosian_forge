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
def _window_adjust(self, m):
    nbytes = m.get_int()
    self.lock.acquire()
    try:
        if self.ultra_debug:
            self._log(DEBUG, 'window up {}'.format(nbytes))
        self.out_window_size += nbytes
        self.out_buffer_cv.notify_all()
    finally:
        self.lock.release()