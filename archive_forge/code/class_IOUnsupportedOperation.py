from contextlib import contextmanager
import curses
from curses import setupterm, tigetnum, tigetstr, tparm
from fcntl import ioctl
from six import text_type, string_types
from os import isatty, environ
import struct
import sys
from termios import TIOCGWINSZ
class IOUnsupportedOperation(Exception):
    """A dummy exception to take the place of Python 3's
        ``io.UnsupportedOperation`` in Python 2"""