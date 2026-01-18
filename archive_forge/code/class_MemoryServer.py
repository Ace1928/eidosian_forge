import contextlib
import errno
import itertools
import os
from io import BytesIO
from stat import S_IFDIR, S_IFLNK, S_IFREG, S_ISDIR
from .. import transport, urlutils
from ..errors import InProcessTransport, LockError, TransportNotPossible
from ..transport import (AppendBasedFileStream, FileExists, LateReadError,
class MemoryServer(transport.Server):
    """Server for the MemoryTransport for testing with."""

    def start_server(self):
        self._dirs = {'/': None}
        self._files = {}
        self._symlinks = {}
        self._locks = {}
        self._scheme = 'memory+%s:///' % id(self)

        def memory_factory(url):
            from . import memory
            result = memory.MemoryTransport(url)
            result._dirs = self._dirs
            result._files = self._files
            result._symlinks = self._symlinks
            result._locks = self._locks
            return result
        self._memory_factory = memory_factory
        transport.register_transport(self._scheme, self._memory_factory)

    def stop_server(self):
        transport.unregister_transport(self._scheme, self._memory_factory)

    def get_url(self):
        """See breezy.transport.Server.get_url."""
        return self._scheme

    def get_bogus_url(self):
        raise NotImplementedError