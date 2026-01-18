import io
import logging
import subprocess
import urllib.parse
from smart_open import utils
class CliRawOutputBase(io.RawIOBase):
    """Writes bytes to HDFS via the "hdfs dfs" command-line interface.

    Implements the io.RawIOBase interface of the standard library.
    """

    def __init__(self, uri):
        self._uri = uri
        self._sub = subprocess.Popen(['hdfs', 'dfs', '-put', '-f', '-', self._uri], stdin=subprocess.PIPE)
        self.raw = None

    def close(self):
        self.flush()
        self._sub.stdin.close()
        self._sub.wait()

    def flush(self):
        self._sub.stdin.flush()

    def writeable(self):
        """Return True if this object is writeable."""
        return self._sub is not None

    def seekable(self):
        """If False, seek(), tell() and truncate() will raise IOError."""
        return False

    def write(self, b):
        self._sub.stdin.write(b)

    def detach(self):
        raise io.UnsupportedOperation('detach() not supported')