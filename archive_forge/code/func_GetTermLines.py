from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import errno
import locale
import os
import struct
import sys
import six
from gslib.utils.constants import WINDOWS_1252
def GetTermLines():
    """Returns number of terminal lines."""
    try:
        import fcntl
        import termios
    except ImportError:
        return _DEFAULT_NUM_TERM_LINES

    def ioctl_GWINSZ(fd):
        try:
            return struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))[0]
        except:
            return 0
    ioc = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not ioc:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            ioc = ioctl_GWINSZ(fd)
            os.close(fd)
        except:
            pass
    if not ioc:
        ioc = os.environ.get('LINES', _DEFAULT_NUM_TERM_LINES)
    return int(ioc)