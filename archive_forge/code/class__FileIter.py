import os, time, mimetypes, zipfile, tarfile
from paste.httpexceptions import (
from paste.httpheaders import (
class _FileIter(object):

    def __init__(self, file, block_size=None, size=None):
        self.file = file
        self.size = size
        self.block_size = block_size or BLOCK_SIZE

    def __iter__(self):
        return self

    def next(self):
        chunk_size = self.block_size
        if self.size is not None:
            if chunk_size > self.size:
                chunk_size = self.size
            self.size -= chunk_size
        data = self.file.read(chunk_size)
        if not data:
            raise StopIteration
        return data
    __next__ = next

    def close(self):
        self.file.close()