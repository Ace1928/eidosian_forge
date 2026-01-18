import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
class FileFileStream(FileStream):
    """A file stream object returned by open_write_stream.

    This version uses a file like object to perform writes.
    """

    def __init__(self, transport, relpath, file_handle):
        FileStream.__init__(self, transport, relpath)
        self.file_handle = file_handle

    def _close(self):
        self.file_handle.close()

    def fdatasync(self):
        """Force data out to physical disk if possible."""
        self.file_handle.flush()
        try:
            fileno = self.file_handle.fileno()
        except AttributeError:
            raise errors.TransportNotPossible()
        osutils.fdatasync(fileno)

    def write(self, bytes):
        osutils.pump_string_file(bytes, self.file_handle)