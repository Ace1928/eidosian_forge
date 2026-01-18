import io
import sys
from gunicorn.http.errors import (NoMoreData, ChunkMissingTerminator,
class EOFReader(object):

    def __init__(self, unreader):
        self.unreader = unreader
        self.buf = io.BytesIO()
        self.finished = False

    def read(self, size):
        if not isinstance(size, int):
            raise TypeError('size must be an integral type')
        if size < 0:
            raise ValueError('Size must be positive.')
        if size == 0:
            return b''
        if self.finished:
            data = self.buf.getvalue()
            ret, rest = (data[:size], data[size:])
            self.buf = io.BytesIO()
            self.buf.write(rest)
            return ret
        data = self.unreader.read()
        while data:
            self.buf.write(data)
            if self.buf.tell() > size:
                break
            data = self.unreader.read()
        if not data:
            self.finished = True
        data = self.buf.getvalue()
        ret, rest = (data[:size], data[size:])
        self.buf = io.BytesIO()
        self.buf.write(rest)
        return ret