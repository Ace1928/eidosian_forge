from io import BytesIO
import mmap
import os
import sys
import zlib
from gitdb.fun import (
from gitdb.util import (
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_bytes
class FlexibleSha1Writer(Sha1Writer):
    """Writer producing a sha1 while passing on the written bytes to the given
    write function"""
    __slots__ = 'writer'

    def __init__(self, writer):
        Sha1Writer.__init__(self)
        self.writer = writer

    def write(self, data):
        Sha1Writer.write(self, data)
        self.writer(data)