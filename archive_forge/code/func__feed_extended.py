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
def _feed_extended(self, m):
    code = m.get_int()
    s = m.get_binary()
    if code != 1:
        self._log(ERROR, 'unknown extended_data type {}; discarding'.format(code))
        return
    if self.combine_stderr:
        self._feed(s)
    else:
        self.in_stderr_buffer.feed(s)