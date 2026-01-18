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
def _pack_names(self):
    pack_files = []
    try:
        dir_contents = self.pack_transport.list_dir('.')
        for name in dir_contents:
            if name.startswith('pack-') and name.endswith('.pack'):
                idx_name = os.path.splitext(name)[0] + '.idx'
                if idx_name in dir_contents:
                    pack_files.append(os.path.splitext(name)[0])
    except TransportNotPossible:
        try:
            f = self.transport.get('info/packs')
        except NoSuchFile:
            warning("No info/packs on remote host;run 'git update-server-info' on remote.")
        else:
            with f:
                pack_files = [os.path.splitext(name)[0] for name in read_packs_file(f)]
    except NoSuchFile:
        pass
    return pack_files