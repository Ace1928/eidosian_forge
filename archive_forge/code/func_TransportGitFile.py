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
def TransportGitFile(transport, filename, mode='rb', bufsize=-1, mask=420):
    if 'a' in mode:
        raise OSError('append mode not supported for Git files')
    if '+' in mode:
        raise OSError('read/write mode not supported for Git files')
    if 'b' not in mode:
        raise OSError('text mode not supported for Git files')
    if 'w' in mode:
        try:
            return _GitFile(transport.local_abspath(filename), mode, bufsize, mask)
        except NotLocalUrl:
            return _RemoteGitFile(transport, filename, mode, bufsize, mask)
    else:
        return transport.get(filename)