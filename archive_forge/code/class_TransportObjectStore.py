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
class TransportObjectStore(PackBasedObjectStore):
    """Git-style object store that exists on disk."""

    def __init__(self, transport, loose_compression_level=-1, pack_compression_level=-1):
        """Open an object store.

        :param transport: Transport to open data from
        """
        super().__init__()
        self.pack_compression_level = pack_compression_level
        self.loose_compression_level = loose_compression_level
        self.transport = transport
        self.pack_transport = self.transport.clone(PACKDIR)
        self._alternates = None

    @classmethod
    def from_config(cls, path, config):
        try:
            default_compression_level = int(config.get((b'core',), b'compression').decode())
        except KeyError:
            default_compression_level = -1
        try:
            loose_compression_level = int(config.get((b'core',), b'looseCompression').decode())
        except KeyError:
            loose_compression_level = default_compression_level
        try:
            pack_compression_level = int(config.get((b'core',), 'packCompression').decode())
        except KeyError:
            pack_compression_level = default_compression_level
        return cls(path, loose_compression_level, pack_compression_level)

    def __eq__(self, other):
        if not isinstance(other, TransportObjectStore):
            return False
        return self.transport == other.transport

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self.transport)

    @property
    def alternates(self):
        if self._alternates is not None:
            return self._alternates
        self._alternates = []
        for path in self._read_alternate_paths():
            t = _mod_transport.get_transport_from_path(path)
            self._alternates.append(self.__class__(t))
        return self._alternates

    def _read_alternate_paths(self):
        try:
            f = self.transport.get('info/alternates')
        except NoSuchFile:
            return []
        ret = []
        with f:
            for l in f.read().splitlines():
                if l[0] == b'#':
                    continue
                if os.path.isabs(l):
                    continue
                ret.append(l)
            return ret

    def _update_pack_cache(self):
        pack_files = set(self._pack_names())
        new_packs = []
        for basename in pack_files:
            pack_name = basename + '.pack'
            if basename not in self._pack_cache:
                try:
                    size = self.pack_transport.stat(pack_name).st_size
                except TransportNotPossible:
                    size = None
                pd = PackData(pack_name, self.pack_transport.get(pack_name), size=size)
                idxname = basename + '.idx'
                idx = load_pack_index_file(idxname, self.pack_transport.get(idxname))
                pack = Pack.from_objects(pd, idx)
                pack._basename = basename
                self._pack_cache[basename] = pack
                new_packs.append(pack)
        for n in set(self._pack_cache) - pack_files:
            self._pack_cache.pop(n).close()
        return new_packs

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

    def _remove_pack(self, pack):
        self.pack_transport.delete(os.path.basename(pack.index.path))
        self.pack_transport.delete(pack.data.filename)
        try:
            del self._pack_cache[os.path.basename(pack._basename)]
        except KeyError:
            pass

    def _iter_loose_objects(self):
        for base in self.transport.list_dir('.'):
            if len(base) != 2:
                continue
            for rest in self.transport.list_dir(base):
                yield (base + rest).encode(sys.getfilesystemencoding())

    def _split_loose_object(self, sha):
        return (sha[:2], sha[2:])

    def _remove_loose_object(self, sha):
        path = osutils.joinpath(self._split_loose_object(sha))
        self.transport.delete(urlutils.quote_from_bytes(path))

    def _get_loose_object(self, sha):
        path = osutils.joinpath(self._split_loose_object(sha))
        try:
            with self.transport.get(urlutils.quote_from_bytes(path)) as f:
                return ShaFile.from_file(f)
        except NoSuchFile:
            return None

    def add_object(self, obj):
        """Add a single object to this object store.

        :param obj: Object to add
        """
        dir, file = self._split_loose_object(obj.id)
        try:
            self.transport.mkdir(urlutils.quote_from_bytes(dir))
        except FileExists:
            pass
        path = urlutils.quote_from_bytes(osutils.pathjoin(dir, file))
        if self.transport.has(path):
            return
        if self.loose_compression_level not in (-1, None):
            raw_string = obj.as_legacy_object(compression_level=self.loose_compression_level)
        else:
            raw_string = obj.as_legacy_object()
        self.transport.put_bytes(path, raw_string)

    @classmethod
    def init(cls, transport):
        try:
            transport.mkdir('info')
        except FileExists:
            pass
        try:
            transport.mkdir(PACKDIR)
        except FileExists:
            pass
        return cls(transport)

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

    def add_thin_pack(self, read_all, read_some, progress=None):
        """Add a new thin pack to this object store.

        Thin packs are packs that contain deltas with parents that exist
        outside the pack. They should never be placed in the object store
        directly, and always indexed and completed as they are copied.

        Args:
          read_all: Read function that blocks until the number of
            requested bytes are read.
          read_some: Read function that returns at least one byte, but may
            not return the number of bytes requested.
        Returns: A Pack object pointing at the now-completed thin pack in the
            objects/pack directory.
        """
        import tempfile
        try:
            dir = self.transport.local_abspath('.')
        except NotLocalUrl:
            f = tempfile.SpooledTemporaryFile(prefix='tmp_pack_')
            path = None
        else:
            f = tempfile.NamedTemporaryFile(dir=dir, prefix='tmp_pack_', delete=False)
            path = f.name
        try:
            indexer = PackIndexer(f, resolve_ext_ref=self.get_raw)
            copier = PackStreamCopier(read_all, read_some, f, delta_iter=indexer)
            copier.verify(progress=progress)
            if f.name:
                os.chmod(f.name, PACK_MODE)
            return self._complete_pack(f, path, len(copier), indexer, progress=progress)
        except BaseException:
            f.close()
            if path:
                os.remove(path)
            raise

    def add_pack(self):
        """Add a new pack to this object store.

        Returns: Fileobject to write to, a commit function to
            call when the pack is finished and an abort
            function.
        """
        import tempfile
        try:
            dir = self.transport.local_abspath('.')
        except NotLocalUrl:
            f = tempfile.SpooledTemporaryFile(prefix='tmp_pack_')
            path = None
        else:
            f = tempfile.NamedTemporaryFile(dir=dir, prefix='tmp_pack_', delete=False)
            path = f.name

        def commit():
            if f.tell() > 0:
                f.seek(0)
                with PackData(path, f) as pd:
                    indexer = PackIndexer.for_pack_data(pd, resolve_ext_ref=self.get_raw)
                    return self._complete_pack(f, path, len(pd), indexer)
            else:
                abort()
                return None

        def abort():
            f.close()
            if path:
                os.remove(path)
        return (f, commit, abort)