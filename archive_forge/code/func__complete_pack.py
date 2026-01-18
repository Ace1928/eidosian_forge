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
def _complete_pack(self, f, path, num_objects, indexer, progress=None):
    """Move a specific file containing a pack into the pack directory.

        Note: The file should be on the same file system as the
            packs directory.

        Args:
          f: Open file object for the pack.
          path: Path to the pack file.
          indexer: A PackIndexer for indexing the pack.
        """
    entries = []
    for i, entry in enumerate(indexer):
        if progress is not None:
            progress(('generating index: %d/%d\r' % (i, num_objects)).encode('ascii'))
        entries.append(entry)
    pack_sha, extra_entries = extend_pack(f, indexer.ext_refs(), get_raw=self.get_raw, compression_level=self.pack_compression_level, progress=progress)
    f.flush()
    try:
        fileno = f.fileno()
    except AttributeError:
        pass
    else:
        os.fsync(fileno)
    entries.extend(extra_entries)
    entries.sort()
    pack_base_name = 'pack-' + iter_sha1((entry[0] for entry in entries)).decode('ascii')
    for pack in self.packs:
        if osutils.basename(pack._basename) == pack_base_name:
            f.close()
            return pack
    target_pack_name = pack_base_name + '.pack'
    target_pack_index = pack_base_name + '.idx'
    if sys.platform == 'win32':
        with suppress(NoSuchFile):
            self.transport.remove(target_pack_name)
    if path:
        f.close()
        self.pack_transport.ensure_base()
        os.rename(path, self.transport.local_abspath(osutils.pathjoin(PACKDIR, target_pack_name)))
    else:
        f.seek(0)
        self.pack_transport.put_file(target_pack_name, f, mode=PACK_MODE)
    with TransportGitFile(self.pack_transport, target_pack_index, 'wb', mask=PACK_MODE) as index_file:
        write_pack_index(index_file, entries, pack_sha)
    final_pack = Pack.from_objects(PackData(target_pack_name, self.pack_transport.get(target_pack_name)), load_pack_index_file(target_pack_index, self.pack_transport.get(target_pack_index)))
    final_pack._basename = pack_base_name
    final_pack.check_length_and_checksum()
    self._add_cached_pack(pack_base_name, final_pack)
    return final_pack