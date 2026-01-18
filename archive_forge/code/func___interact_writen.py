import os
import sys
import time
import pty
import tty
import errno
import signal
from contextlib import contextmanager
import ptyprocess
from ptyprocess.ptyprocess import use_native_pty_fork
from .exceptions import ExceptionPexpect, EOF, TIMEOUT
from .spawnbase import SpawnBase
from .utils import (
def __interact_writen(self, fd, data):
    """This is used by the interact() method.
        """
    while data != b'' and self.isalive():
        n = os.write(fd, data)
        data = data[n:]