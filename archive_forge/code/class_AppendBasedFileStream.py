import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
class AppendBasedFileStream(FileStream):
    """A file stream object returned by open_write_stream.

    This version uses append on a transport to perform writes.
    """

    def write(self, bytes):
        self.transport.append_bytes(self.relpath, bytes)