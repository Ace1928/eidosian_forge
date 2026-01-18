import os
import stat
import sys
import warnings
from contextlib import suppress
from io import BytesIO
from typing import (
from .errors import NotTreeError
from .file import GitFile
from .objects import (
from .pack import (
from .protocol import DEPTH_INFINITE
from .refs import PEELED_TAG_SUFFIX, Ref
class DiskObjectStore(PackBasedObjectStore):
    """Git-style object store that exists on disk."""

    def __init__(self, path, loose_compression_level=-1, pack_compression_level=-1) -> None:
        """Open an object store.

        Args:
          path: Path of the object store.
          loose_compression_level: zlib compression level for loose objects
          pack_compression_level: zlib compression level for pack objects
        """
        super().__init__(pack_compression_level=pack_compression_level)
        self.path = path
        self.pack_dir = os.path.join(self.path, PACKDIR)
        self._alternates = None
        self.loose_compression_level = loose_compression_level
        self.pack_compression_level = pack_compression_level

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}({self.path!r})>'

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

    @property
    def alternates(self):
        if self._alternates is not None:
            return self._alternates
        self._alternates = []
        for path in self._read_alternate_paths():
            self._alternates.append(DiskObjectStore(path))
        return self._alternates

    def _read_alternate_paths(self):
        try:
            f = GitFile(os.path.join(self.path, INFODIR, 'alternates'), 'rb')
        except FileNotFoundError:
            return
        with f:
            for line in f.readlines():
                line = line.rstrip(b'\n')
                if line.startswith(b'#'):
                    continue
                if os.path.isabs(line):
                    yield os.fsdecode(line)
                else:
                    yield os.fsdecode(os.path.join(os.fsencode(self.path), line))

    def add_alternate_path(self, path):
        """Add an alternate path to this object store."""
        try:
            os.mkdir(os.path.join(self.path, INFODIR))
        except FileExistsError:
            pass
        alternates_path = os.path.join(self.path, INFODIR, 'alternates')
        with GitFile(alternates_path, 'wb') as f:
            try:
                orig_f = open(alternates_path, 'rb')
            except FileNotFoundError:
                pass
            else:
                with orig_f:
                    f.write(orig_f.read())
            f.write(os.fsencode(path) + b'\n')
        if not os.path.isabs(path):
            path = os.path.join(self.path, path)
        self.alternates.append(DiskObjectStore(path))

    def _update_pack_cache(self):
        """Read and iterate over new pack files and cache them."""
        try:
            pack_dir_contents = os.listdir(self.pack_dir)
        except FileNotFoundError:
            self.close()
            return []
        pack_files = set()
        for name in pack_dir_contents:
            if name.startswith('pack-') and name.endswith('.pack'):
                idx_name = os.path.splitext(name)[0] + '.idx'
                if idx_name in pack_dir_contents:
                    pack_name = name[:-len('.pack')]
                    pack_files.add(pack_name)
        new_packs = []
        for f in pack_files:
            if f not in self._pack_cache:
                pack = Pack(os.path.join(self.pack_dir, f))
                new_packs.append(pack)
                self._pack_cache[f] = pack
        for f in set(self._pack_cache) - pack_files:
            self._pack_cache.pop(f).close()
        return new_packs

    def _get_shafile_path(self, sha):
        return hex_to_filename(self.path, sha)

    def _iter_loose_objects(self):
        for base in os.listdir(self.path):
            if len(base) != 2:
                continue
            for rest in os.listdir(os.path.join(self.path, base)):
                sha = os.fsencode(base + rest)
                if not valid_hexsha(sha):
                    continue
                yield sha

    def _get_loose_object(self, sha):
        path = self._get_shafile_path(sha)
        try:
            return ShaFile.from_path(path)
        except FileNotFoundError:
            return None

    def _remove_loose_object(self, sha):
        os.remove(self._get_shafile_path(sha))

    def _remove_pack(self, pack):
        try:
            del self._pack_cache[os.path.basename(pack._basename)]
        except KeyError:
            pass
        pack.close()
        os.remove(pack.data.path)
        os.remove(pack.index.path)

    def _get_pack_basepath(self, entries):
        suffix = iter_sha1((entry[0] for entry in entries))
        suffix = suffix.decode('ascii')
        return os.path.join(self.pack_dir, 'pack-' + suffix)

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
        f.close()
        entries.extend(extra_entries)
        entries.sort()
        pack_base_name = self._get_pack_basepath(entries)
        for pack in self.packs:
            if pack._basename == pack_base_name:
                return pack
        target_pack_path = pack_base_name + '.pack'
        target_index_path = pack_base_name + '.idx'
        if sys.platform == 'win32':
            with suppress(FileNotFoundError):
                os.remove(target_pack_path)
        os.rename(path, target_pack_path)
        with GitFile(target_index_path, 'wb', mask=PACK_MODE) as index_file:
            write_pack_index(index_file, entries, pack_sha)
        final_pack = Pack(pack_base_name)
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
        fd, path = tempfile.mkstemp(dir=self.path, prefix='tmp_pack_')
        with os.fdopen(fd, 'w+b') as f:
            os.chmod(path, PACK_MODE)
            indexer = PackIndexer(f, resolve_ext_ref=self.get_raw)
            copier = PackStreamCopier(read_all, read_some, f, delta_iter=indexer)
            copier.verify(progress=progress)
            return self._complete_pack(f, path, len(copier), indexer, progress=progress)

    def add_pack(self):
        """Add a new pack to this object store.

        Returns: Fileobject to write to, a commit function to
            call when the pack is finished and an abort
            function.
        """
        import tempfile
        fd, path = tempfile.mkstemp(dir=self.pack_dir, suffix='.pack')
        f = os.fdopen(fd, 'w+b')
        os.chmod(path, PACK_MODE)

        def commit():
            if f.tell() > 0:
                f.seek(0)
                with PackData(path, f) as pd:
                    indexer = PackIndexer.for_pack_data(pd, resolve_ext_ref=self.get_raw)
                    return self._complete_pack(f, path, len(pd), indexer)
            else:
                f.close()
                os.remove(path)
                return None

        def abort():
            f.close()
            os.remove(path)
        return (f, commit, abort)

    def add_object(self, obj):
        """Add a single object to this object store.

        Args:
          obj: Object to add
        """
        path = self._get_shafile_path(obj.id)
        dir = os.path.dirname(path)
        try:
            os.mkdir(dir)
        except FileExistsError:
            pass
        if os.path.exists(path):
            return
        with GitFile(path, 'wb', mask=PACK_MODE) as f:
            f.write(obj.as_legacy_object(compression_level=self.loose_compression_level))

    @classmethod
    def init(cls, path):
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        os.mkdir(os.path.join(path, 'info'))
        os.mkdir(os.path.join(path, PACKDIR))
        return cls(path)