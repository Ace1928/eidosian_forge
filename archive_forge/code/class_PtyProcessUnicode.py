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
class PtyProcessUnicode(PtyProcess):
    """Unicode wrapper around a process running in a pseudoterminal.

    This class exposes a similar interface to :class:`PtyProcess`, but its read
    methods return unicode, and its :meth:`write` accepts unicode.
    """
    if PY3:
        string_type = str
    else:
        string_type = unicode

    def __init__(self, pid, fd, encoding='utf-8', codec_errors='strict'):
        super(PtyProcessUnicode, self).__init__(pid, fd)
        self.encoding = encoding
        self.codec_errors = codec_errors
        self.decoder = codecs.getincrementaldecoder(encoding)(errors=codec_errors)

    def read(self, size=1024):
        """Read at most ``size`` bytes from the pty, return them as unicode.

        Can block if there is nothing to read. Raises :exc:`EOFError` if the
        terminal was closed.

        The size argument still refers to bytes, not unicode code points.
        """
        b = super(PtyProcessUnicode, self).read(size)
        return self.decoder.decode(b, final=False)

    def readline(self):
        """Read one line from the pseudoterminal, and return it as unicode.

        Can block if there is nothing to read. Raises :exc:`EOFError` if the
        terminal was closed.
        """
        b = super(PtyProcessUnicode, self).readline()
        return self.decoder.decode(b, final=False)

    def write(self, s):
        """Write the unicode string ``s`` to the pseudoterminal.

        Returns the number of bytes written.
        """
        b = s.encode(self.encoding)
        return super(PtyProcessUnicode, self).write(b)