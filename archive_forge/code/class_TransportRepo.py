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
class TransportRepo(BaseRepo):

    def __init__(self, transport, bare, refs_text=None):
        self.transport = transport
        self.bare = bare
        try:
            with transport.get(CONTROLDIR) as f:
                path = read_gitfile(f)
        except (ReadError, NoSuchFile):
            if self.bare:
                self._controltransport = self.transport
            else:
                self._controltransport = self.transport.clone('.git')
        else:
            self._controltransport = self.transport.clone(urlutils.quote_from_bytes(path))
        commondir = self.get_named_file(COMMONDIR)
        if commondir is not None:
            with commondir:
                commondir = os.path.join(self.controldir(), commondir.read().rstrip(b'\r\n').decode(sys.getfilesystemencoding()))
                self._commontransport = _mod_transport.get_transport_from_path(commondir)
        else:
            self._commontransport = self._controltransport
        config = self.get_config()
        object_store = TransportObjectStore.from_config(self._commontransport.clone(OBJECTDIR), config)
        if refs_text is not None:
            refs_container = InfoRefsContainer(BytesIO(refs_text))
            try:
                head = TransportRefsContainer(self._commontransport).read_loose_ref(b'HEAD')
            except KeyError:
                pass
            else:
                refs_container._refs[b'HEAD'] = head
        else:
            refs_container = TransportRefsContainer(self._commontransport, self._controltransport)
        super().__init__(object_store, refs_container)

    def controldir(self):
        return self._controltransport.local_abspath('.')

    def commondir(self):
        return self._commontransport.local_abspath('.')

    def close(self):
        """Close any files opened by this repository."""
        self.object_store.close()

    @property
    def path(self):
        return self.transport.local_abspath('.')

    def _determine_file_mode(self):
        if sys.platform == 'win32':
            return False
        return True

    def _determine_symlinks(self):
        try:
            return osutils.supports_symlinks(self.path)
        except NotLocalUrl:
            return sys.platform != 'win32'

    def get_named_file(self, path):
        """Get a file from the control dir with a specific name.

        Although the filename should be interpreted as a filename relative to
        the control dir in a disk-baked Repo, the object returned need not be
        pointing to a file in that location.

        :param path: The path to the file, relative to the control dir.
        :return: An open file object, or None if the file does not exist.
        """
        try:
            return self._controltransport.get(path.lstrip('/'))
        except NoSuchFile:
            return None

    def _put_named_file(self, relpath, contents):
        self._controltransport.put_bytes(relpath, contents)

    def index_path(self):
        """Return the path to the index file."""
        return self._controltransport.local_abspath(INDEX_FILENAME)

    def open_index(self):
        """Open the index for this repository."""
        from dulwich.index import Index
        if not self.has_index():
            raise NoIndexPresent()
        return Index(self.index_path())

    def has_index(self):
        """Check if an index is present."""
        return not self.bare

    def get_config(self):
        from dulwich.config import ConfigFile
        try:
            with self._controltransport.get('config') as f:
                return ConfigFile.from_file(f)
        except NoSuchFile:
            return ConfigFile()

    def get_config_stack(self):
        from dulwich.config import StackedConfig
        backends = []
        p = self.get_config()
        if p is not None:
            backends.append(p)
            writable = p
        else:
            writable = None
        backends.extend(StackedConfig.default_backends())
        return StackedConfig(backends, writable=writable)

    def __repr__(self):
        return '<{} for {!r}>'.format(self.__class__.__name__, self.transport)

    @classmethod
    def init(cls, transport, bare=False):
        if not bare:
            try:
                transport.mkdir('.git')
            except FileExists:
                raise AlreadyControlDirError(transport.base)
            control_transport = transport.clone('.git')
        else:
            control_transport = transport
        for d in BASE_DIRECTORIES:
            try:
                control_transport.mkdir('/'.join(d))
            except FileExists:
                pass
        try:
            control_transport.mkdir(OBJECTDIR)
        except FileExists:
            raise AlreadyControlDirError(transport.base)
        TransportObjectStore.init(control_transport.clone(OBJECTDIR))
        ret = cls(transport, bare)
        ret.refs.set_symbolic_ref(b'HEAD', b'refs/heads/master')
        ret._init_files(bare)
        return ret