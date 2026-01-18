import json
import os
import posixpath
import stat
import sys
import tempfile
import urllib.parse as urlparse
import zlib
from configparser import ConfigParser
from io import BytesIO
from geventhttpclient import HTTPClient
from ..greenthreads import GreenThreadsMissingObjectFinder
from ..lru_cache import LRUSizeCache
from ..object_store import INFODIR, PACKDIR, PackBasedObjectStore
from ..objects import S_ISGITLINK, Blob, Commit, Tag, Tree
from ..pack import (
from ..protocol import TCP_GIT_PORT
from ..refs import InfoRefsContainer, read_info_refs, write_info_refs
from ..repo import OBJECTDIR, BaseRepo
from ..server import Backend, TCPGitServer
class SwiftPackReader:
    """A SwiftPackReader that mimic read and sync method.

    The reader allows to read a specified amount of bytes from
    a given offset of a Swift object. A read offset is kept internally.
    The reader will read from Swift a specified amount of data to complete
    its internal buffer. chunk_length specify the amount of data
    to read from Swift.
    """

    def __init__(self, scon, filename, pack_length) -> None:
        """Initialize a SwiftPackReader.

        Args:
          scon: a `SwiftConnector` instance
          filename: the pack filename
          pack_length: The size of the pack object
        """
        self.scon = scon
        self.filename = filename
        self.pack_length = pack_length
        self.offset = 0
        self.base_offset = 0
        self.buff = b''
        self.buff_length = self.scon.chunk_length

    def _read(self, more=False):
        if more:
            self.buff_length = self.buff_length * 2
        offset = self.base_offset
        r = min(self.base_offset + self.buff_length, self.pack_length)
        ret = self.scon.get_object(self.filename, range=f'{offset}-{r}')
        self.buff = ret

    def read(self, length):
        """Read a specified amount of Bytes form the pack object.

        Args:
          length: amount of bytes to read
        Returns:
          a bytestring
        """
        end = self.offset + length
        if self.base_offset + end > self.pack_length:
            data = self.buff[self.offset:]
            self.offset = end
            return data
        if end > len(self.buff):
            self._read(more=True)
            return self.read(length)
        data = self.buff[self.offset:end]
        self.offset = end
        return data

    def seek(self, offset):
        """Seek to a specified offset.

        Args:
          offset: the offset to seek to
        """
        self.base_offset = offset
        self._read()
        self.offset = 0

    def read_checksum(self):
        """Read the checksum from the pack.

        Returns: the checksum bytestring
        """
        return self.scon.get_object(self.filename, range='-20')