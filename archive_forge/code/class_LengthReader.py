import io
import sys
from gunicorn.http.errors import (NoMoreData, ChunkMissingTerminator,
class LengthReader(object):

    def __init__(self, unreader, length):
        self.unreader = unreader
        self.length = length

    def read(self, size):
        if not isinstance(size, int):
            raise TypeError('size must be an integral type')
        size = min(self.length, size)
        if size < 0:
            raise ValueError('Size must be positive.')
        if size == 0:
            return b''
        buf = io.BytesIO()
        data = self.unreader.read()
        while data:
            buf.write(data)
            if buf.tell() >= size:
                break
            data = self.unreader.read()
        buf = buf.getvalue()
        ret, rest = (buf[:size], buf[size:])
        self.unreader.unread(rest)
        self.length -= size
        return ret