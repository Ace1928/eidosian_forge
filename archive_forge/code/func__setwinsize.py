import codecs
import errno
import fcntl
import io
import os
import pty
import resource
import signal
import struct
import sys
import termios
import time
from pty import (STDIN_FILENO, CHILD)
from .util import which, PtyProcessError
def _setwinsize(fd, rows, cols):
    TIOCSWINSZ = getattr(termios, 'TIOCSWINSZ', -2146929561)
    s = struct.pack('HHHH', rows, cols, 0, 0)
    fcntl.ioctl(fd, TIOCSWINSZ, s)