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
class SwiftObjectStore(PackBasedObjectStore):
    """A Swift Object Store.

    Allow to manage a bare Git repository from Openstack Swift.
    This object store only supports pack files and not loose objects.
    """

    def __init__(self, scon) -> None:
        """Open a Swift object store.

        Args:
          scon: A `SwiftConnector` instance
        """
        super().__init__()
        self.scon = scon
        self.root = self.scon.root
        self.pack_dir = posixpath.join(OBJECTDIR, PACKDIR)
        self._alternates = None

    def _update_pack_cache(self):
        objects = self.scon.get_container_objects()
        pack_files = [o['name'].replace('.pack', '') for o in objects if o['name'].endswith('.pack')]
        ret = []
        for basename in pack_files:
            pack = SwiftPack(basename, scon=self.scon)
            self._pack_cache[basename] = pack
            ret.append(pack)
        return ret

    def _iter_loose_objects(self):
        """Loose objects are not supported by this repository."""
        return []

    def pack_info_get(self, sha):
        for pack in self.packs:
            if sha in pack:
                return pack.pack_info[sha]

    def _collect_ancestors(self, heads, common=set()):

        def _find_parents(commit):
            for pack in self.packs:
                if commit in pack:
                    try:
                        parents = pack.pack_info[commit][1]
                    except KeyError:
                        return []
                    return parents
        bases = set()
        commits = set()
        queue = []
        queue.extend(heads)
        while queue:
            e = queue.pop(0)
            if e in common:
                bases.add(e)
            elif e not in commits:
                commits.add(e)
                parents = _find_parents(e)
                queue.extend(parents)
        return (commits, bases)

    def add_pack(self):
        """Add a new pack to this object store.

        Returns: Fileobject to write to and a commit function to
            call when the pack is finished.
        """
        f = BytesIO()

        def commit():
            f.seek(0)
            pack = PackData(file=f, filename='')
            entries = pack.sorted_entries()
            if entries:
                basename = posixpath.join(self.pack_dir, 'pack-%s' % iter_sha1((entry[0] for entry in entries)))
                index = BytesIO()
                write_pack_index_v2(index, entries, pack.get_stored_checksum())
                self.scon.put_object(basename + '.pack', f)
                f.close()
                self.scon.put_object(basename + '.idx', index)
                index.close()
                final_pack = SwiftPack(basename, scon=self.scon)
                final_pack.check_length_and_checksum()
                self._add_cached_pack(basename, final_pack)
                return final_pack
            else:
                return None

        def abort():
            pass
        return (f, commit, abort)

    def add_object(self, obj):
        self.add_objects([(obj, None)])

    def _pack_cache_stale(self):
        return False

    def _get_loose_object(self, sha):
        return None

    def add_thin_pack(self, read_all, read_some):
        """Read a thin pack.

        Read it from a stream and complete it in a temporary file.
        Then the pack and the corresponding index file are uploaded to Swift.
        """
        fd, path = tempfile.mkstemp(prefix='tmp_pack_')
        f = os.fdopen(fd, 'w+b')
        try:
            indexer = PackIndexer(f, resolve_ext_ref=self.get_raw)
            copier = PackStreamCopier(read_all, read_some, f, delta_iter=indexer)
            copier.verify()
            return self._complete_thin_pack(f, path, copier, indexer)
        finally:
            f.close()
            os.unlink(path)

    def _complete_thin_pack(self, f, path, copier, indexer):
        entries = list(indexer)
        f.seek(0)
        write_pack_header(f, len(entries) + len(indexer.ext_refs()))
        f.flush()
        new_sha = compute_file_sha(f, end_ofs=-20)
        f.seek(0, os.SEEK_CUR)
        for ext_sha in indexer.ext_refs():
            assert len(ext_sha) == 20
            type_num, data = self.get_raw(ext_sha)
            offset = f.tell()
            crc32 = write_pack_object(f, type_num, data, sha=new_sha)
            entries.append((ext_sha, offset, crc32))
        pack_sha = new_sha.digest()
        f.write(pack_sha)
        f.flush()
        entries.sort()
        pack_base_name = posixpath.join(self.pack_dir, 'pack-' + os.fsdecode(iter_sha1((e[0] for e in entries))))
        self.scon.put_object(pack_base_name + '.pack', f)
        filename = pack_base_name + '.idx'
        index_file = BytesIO()
        write_pack_index_v2(index_file, entries, pack_sha)
        self.scon.put_object(filename, index_file)
        f.seek(0)
        pack_data = PackData(filename='', file=f)
        index_file.seek(0)
        pack_index = load_pack_index_file('', index_file)
        serialized_pack_info = pack_info_create(pack_data, pack_index)
        f.close()
        index_file.close()
        pack_info_file = BytesIO(serialized_pack_info)
        filename = pack_base_name + '.info'
        self.scon.put_object(filename, pack_info_file)
        pack_info_file.close()
        final_pack = SwiftPack(pack_base_name, scon=self.scon)
        final_pack.check_length_and_checksum()
        self._add_cached_pack(pack_base_name, final_pack)
        return final_pack