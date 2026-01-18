import cgi
import email.utils as email_utils
import http.client as http_client
import os
from io import BytesIO
from ... import errors, osutils
class ResponseFile:
    """A wrapper around the http socket containing the result of a GET request.

    Only read() and seek() (forward) are supported.

    """

    def __init__(self, path, infile):
        """Constructor.

        :param path: File url, for error reports.

        :param infile: File-like socket set at body start.
        """
        self._path = path
        self._file = infile
        self._pos = 0

    def close(self):
        """Close this file.

        Dummy implementation for consistency with the 'file' API.
        """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def read(self, size=None):
        """Read size bytes from the current position in the file.

        :param size:  The number of bytes to read.  Leave unspecified or pass
            -1 to read to EOF.
        """
        data = self._file.read(size)
        self._pos += len(data)
        return data

    def readline(self):
        data = self._file.readline()
        self._pos += len(data)
        return data

    def readlines(self, size=None):
        data = self._file.readlines()
        self._pos += sum(map(len, data))
        return data

    def __iter__(self):
        while True:
            line = self.readline()
            if not line:
                return
            yield line

    def tell(self):
        return self._pos

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            if offset < self._pos:
                raise AssertionError("Can't seek backwards, pos: %s, offset: %s" % (self._pos, offset))
            to_discard = offset - self._pos
        elif whence == os.SEEK_CUR:
            to_discard = offset
        else:
            raise AssertionError("Can't seek backwards")
        if to_discard:
            self.read(to_discard)