import os
import posixpath
import sys
from io import BytesIO
from dulwich.errors import NoIndexPresent
from dulwich.file import FileLocked, _GitFile
from dulwich.object_store import (PACK_MODE, PACKDIR, PackBasedObjectStore,
from dulwich.objects import ShaFile
from dulwich.pack import (PACK_SPOOL_FILE_MAX_SIZE, MemoryPackIndex, Pack,
from dulwich.refs import SymrefLoop
from dulwich.repo import (BASE_DIRECTORIES, COMMONDIR, CONTROLDIR,
from .. import osutils
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..errors import (AlreadyControlDirError, LockBroken, LockContention,
from ..lock import LogicalLockResult
from ..trace import warning
from ..transport import FileExists, NoSuchFile
from ..transport.local import LocalTransport
class _RemoteGitFile(object):

    def __init__(self, transport, filename, mode, bufsize, mask):
        self.transport = transport
        self.filename = filename
        self.mode = mode
        self.bufsize = bufsize
        self.mask = mask
        import tempfile
        self._file = tempfile.SpooledTemporaryFile(max_size=1024 * 1024)
        self._closed = False
        for method in _GitFile.PROXY_METHODS:
            setattr(self, method, getattr(self._file, method))

    def abort(self):
        self._file.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.abort()
        else:
            self.close()

    def __getattr__(self, name):
        """Proxy property calls to the underlying file."""
        if name in _GitFile.PROXY_PROPERTIES:
            return getattr(self._file, name)
        raise AttributeError(name)

    def close(self):
        if self._closed:
            return
        self._file.flush()
        self._file.seek(0)
        self.transport.put_file(self.filename, self._file)
        self._file.close()
        self._closed = True