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
@staticmethod
def _coerce_read_string(s):
    return s