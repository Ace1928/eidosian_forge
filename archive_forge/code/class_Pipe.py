import fcntl
import os
from functools import partial
from pyudev._ctypeslib.libc import ERROR_CHECKERS, FD_PAIR, SIGNATURES
from pyudev._ctypeslib.utils import load_ctypes_library
class Pipe:
    """A unix pipe.

    A pipe object provides two file objects: :attr:`source` is a readable file
    object, and :attr:`sink` a writeable.  Bytes written to :attr:`sink` appear
    at :attr:`source`.

    Open a pipe with :meth:`open()`.

    """

    @classmethod
    def open(cls):
        """Open and return a new :class:`Pipe`.

        The pipe uses non-blocking IO."""
        source, sink = _PIPE2(os.O_NONBLOCK | O_CLOEXEC)
        return cls(source, sink)

    def __init__(self, source_fd, sink_fd):
        """Create a new pipe object from the given file descriptors.

        ``source_fd`` is a file descriptor for the readable side of the pipe,
        ``sink_fd`` is a file descriptor for the writeable side."""
        self.source = os.fdopen(source_fd, 'rb', 0)
        self.sink = os.fdopen(sink_fd, 'wb', 0)

    def close(self):
        """Closes both sides of the pipe."""
        try:
            self.source.close()
        finally:
            self.sink.close()