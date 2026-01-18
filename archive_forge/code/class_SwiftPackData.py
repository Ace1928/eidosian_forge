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
class SwiftPackData(PackData):
    """The data contained in a packfile.

    We use the SwiftPackReader to read bytes from packs stored in Swift
    using the Range header feature of Swift.
    """

    def __init__(self, scon, filename) -> None:
        """Initialize a SwiftPackReader.

        Args:
          scon: a `SwiftConnector` instance
          filename: the pack filename
        """
        self.scon = scon
        self._filename = filename
        self._header_size = 12
        headers = self.scon.get_object_stat(self._filename)
        self.pack_length = int(headers['content-length'])
        pack_reader = SwiftPackReader(self.scon, self._filename, self.pack_length)
        version, self._num_objects = read_pack_header(pack_reader.read)
        self._offset_cache = LRUSizeCache(1024 * 1024 * self.scon.cache_length, compute_size=_compute_object_size)
        self.pack = None

    def get_object_at(self, offset):
        if offset in self._offset_cache:
            return self._offset_cache[offset]
        assert offset >= self._header_size
        pack_reader = SwiftPackReader(self.scon, self._filename, self.pack_length)
        pack_reader.seek(offset)
        unpacked, _ = unpack_object(pack_reader.read)
        return (unpacked.pack_type_num, unpacked._obj())

    def get_stored_checksum(self):
        pack_reader = SwiftPackReader(self.scon, self._filename, self.pack_length)
        return pack_reader.read_checksum()

    def close(self):
        pass