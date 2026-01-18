import posixpath
import stat
import struct
import tarfile
from contextlib import closing
from io import BytesIO
from os import SEEK_END
class ChunkedBytesIO:
    """Turn a list of bytestrings into a file-like object.

    This is similar to creating a `BytesIO` from a concatenation of the
    bytestring list, but saves memory by NOT creating one giant bytestring
    first::

        BytesIO(b''.join(list_of_bytestrings)) =~= ChunkedBytesIO(
            list_of_bytestrings)
    """

    def __init__(self, contents) -> None:
        self.contents = contents
        self.pos = (0, 0)

    def read(self, maxbytes=None):
        if maxbytes < 0:
            maxbytes = float('inf')
        buf = []
        chunk, cursor = self.pos
        while chunk < len(self.contents):
            if maxbytes < len(self.contents[chunk]) - cursor:
                buf.append(self.contents[chunk][cursor:cursor + maxbytes])
                cursor += maxbytes
                self.pos = (chunk, cursor)
                break
            else:
                buf.append(self.contents[chunk][cursor:])
                maxbytes -= len(self.contents[chunk]) - cursor
                chunk += 1
                cursor = 0
                self.pos = (chunk, cursor)
        return b''.join(buf)