from contextlib import contextmanager
import curses
from curses import setupterm, tigetnum, tigetstr, tparm
from fcntl import ioctl
from six import text_type, string_types
from os import isatty, environ
import struct
import sys
from termios import TIOCGWINSZ
def _height_and_width(self):
    """Return a tuple of (terminal height, terminal width).

        Start by trying TIOCGWINSZ (Terminal I/O-Control: Get Window Size),
        falling back to environment variables (LINES, COLUMNS), and returning
        (None, None) if those are unavailable or invalid.

        """
    for descriptor in (self._init_descriptor, sys.__stdout__):
        try:
            return struct.unpack('hhhh', ioctl(descriptor, TIOCGWINSZ, '\x00' * 8))[0:2]
        except IOError:
            pass
    try:
        return (int(environ.get('LINES')), int(environ.get('COLUMNS')))
    except TypeError:
        return (None, None)