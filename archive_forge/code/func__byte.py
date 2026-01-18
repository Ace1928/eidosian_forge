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
def _byte(i):
    return chr(i)