from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
def _GetTermSizePosix():
    """Returns the Posix terminal x and y dimensions."""
    import fcntl
    import struct
    import termios

    def _GetXY(fd):
        """Returns the terminal (x,y) size for fd.

    Args:
      fd: The terminal file descriptor.

    Returns:
      The terminal (x,y) size for fd or None on error.
    """
        try:
            rc = struct.unpack(b'hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, 'junk'))
            return (rc[1], rc[0]) if rc else None
        except:
            return None
    xy = _GetXY(0) or _GetXY(1) or _GetXY(2)
    if not xy:
        fd = None
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            xy = _GetXY(fd)
        except:
            xy = None
        finally:
            if fd is not None:
                os.close(fd)
    return xy