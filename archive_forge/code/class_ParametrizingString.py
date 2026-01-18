from contextlib import contextmanager
import curses
from curses import setupterm, tigetnum, tigetstr, tparm
from fcntl import ioctl
from six import text_type, string_types
from os import isatty, environ
import struct
import sys
from termios import TIOCGWINSZ
class ParametrizingString(text_type):
    """A Unicode string which can be called to parametrize it as a terminal
    capability"""

    def __new__(cls, formatting, normal=None):
        """Instantiate.

        :arg normal: If non-None, indicates that, once parametrized, this can
            be used as a ``FormattingString``. The value is used as the
            "normal" capability.

        """
        new = text_type.__new__(cls, formatting)
        new._normal = normal
        return new

    def __call__(self, *args):
        try:
            parametrized = tparm(self.encode('latin1'), *args).decode('latin1')
            return parametrized if self._normal is None else FormattingString(parametrized, self._normal)
        except curses.error:
            return u''
        except TypeError:
            if len(args) == 1 and isinstance(args[0], string_types):
                raise TypeError('A native or nonexistent capability template received %r when it was expecting ints. You probably misspelled a formatting call like bright_red_on_white(...).' % args)
            else:
                raise